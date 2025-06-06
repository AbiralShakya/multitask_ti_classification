import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from collections import defaultdict

class CrystalGraphConv(MessagePassing):
    """Crystal Graph Convolutional Layer"""
    def __init__(self, node_dim, edge_dim):
        super(CrystalGraphConv, self).__init__(aggr='add')
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Linear layers for node and edge features
        self.node_linear = nn.Linear(node_dim, node_dim)
        self.edge_linear = nn.Linear(edge_dim, node_dim)
        self.gate_linear = nn.Linear(2 * node_dim, node_dim)
        
        self.batch_norm = nn.BatchNorm1d(node_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_attr):
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # Transform edge features
        edge_features = self.edge_linear(edge_attr)
        # Combine node and edge features
        combined = torch.cat([x_j, edge_features], dim=1)
        gate = self.activation(self.gate_linear(combined))
        return gate * x_j
    
    def update(self, aggr_out, x):
        # Update node features
        out = self.node_linear(x) + aggr_out
        return self.batch_norm(out)

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for crystal structure representation"""
    def __init__(self, node_dim=64, edge_dim=32, hidden_dim=128, num_layers=3):
        super(GraphNeuralNetwork, self).__init__()
        
        self.node_embedding = nn.Linear(27 + 50, node_dim)  # Element features
        self.edge_embedding = nn.Linear(1, edge_dim)  # Distance features
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList([
            CrystalGraphConv(node_dim, edge_dim) for _ in range(num_layers)
        ])
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed features
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final projection
        x = self.final_projection(x)
        
        return x

class PersistentHomologyProcessor:
    """Atom-Specific Persistent Homology feature extraction"""
    def __init__(self, cutoff_radius=8.0):
        self.cutoff_radius = cutoff_radius
        
    def compute_distance_matrix(self, positions):
        """Compute pairwise distance matrix"""
        return squareform(pdist(positions))
    
    def compute_persistent_homology(self, positions, max_dimension=2):
        """Compute persistent homology features"""
        distance_matrix = self.compute_distance_matrix(positions)
        
        # Compute persistent homology using Ripser
        result = ripser(distance_matrix, maxdim=max_dimension, distance_matrix=True)
        
        features = []
        
        # Extract features from each homological dimension
        for dim in range(max_dimension + 1):
            if len(result['dgms'][dim]) > 0:
                diagrams = result['dgms'][dim]
                birth_death = diagrams[diagrams[:, 1] != np.inf]  # Remove infinite bars
                
                if len(birth_death) > 0:
                    birth = birth_death[:, 0]
                    death = birth_death[:, 1]
                    persistence = death - birth
                    
                    # Statistical features
                    stats = [
                        np.min(birth), np.max(birth), np.mean(birth), np.std(birth), np.sum(birth),
                        np.min(death), np.max(death), np.mean(death), np.std(death), np.sum(death),
                        np.min(persistence), np.max(persistence), np.mean(persistence), 
                        np.std(persistence), np.sum(persistence)
                    ]
                else:
                    stats = [0.0] * 15
            else:
                stats = [0.0] * 15
            
            features.extend(stats)
        
        return np.array(features)
    
    def extract_atom_specific_features(self, positions, elements):
        """Extract atom-specific persistent homology features"""
        unique_elements = np.unique(elements)
        all_features = []
        
        # For each element type
        for element in unique_elements:
            # Get positions of atoms of this element type
            element_mask = elements == element
            element_positions = positions[element_mask]
            
            if len(element_positions) > 0:
                # Compute features for this element type
                features = self.compute_persistent_homology(element_positions)
                all_features.extend(features)
        
        # Pad or truncate to fixed size (simplified approach)
        target_size = 3115  # As mentioned in the paper
        if len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        else:
            all_features = all_features[:target_size]
        
        return np.array(all_features)

class TopologicalClassifier(nn.Module):
    """Combined GNN + Persistent Homology classifier"""
    def __init__(self, gnn_output_dim=128, ph_input_dim=3115, num_classes=2):
        super(TopologicalClassifier, self).__init__()
        
        # Graph Neural Network
        self.gnn = GraphNeuralNetwork()
        
        # Persistent Homology feature processor
        self.ph_processor = nn.Sequential(
            nn.Linear(ph_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Combined classifier
        combined_dim = gnn_output_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, graph_data, ph_features):
        # Extract graph features
        graph_features = self.gnn(graph_data)
        
        # Process persistent homology features
        ph_features = self.ph_processor(ph_features)
        
        # Combine features
        combined_features = torch.cat([graph_features, ph_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output

class DataPreprocessor:
    """Data preprocessing utilities"""
    def __init__(self):
        # Element feature mapping (simplified)
        self.element_features = self._create_element_features()
        
    def _create_element_features(self):
        """Create one-hot encoded element features"""
        # Simplified element features (in practice, would use full periodic table data)
        elements = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                   'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        
        features = {}
        for i, element in enumerate(elements):
            # One-hot encoding for element type
            one_hot = np.zeros(len(elements))
            one_hot[i] = 1
            
            # Additional chemical properties (simplified)
            properties = np.random.rand(50)  # Placeholder for real chemical properties
            
            features[element] = np.concatenate([one_hot, properties])
        
        return features
    
    def crystal_to_graph(self, positions, elements, cell_parameters=None):
        """Convert crystal structure to graph representation"""
        num_atoms = len(positions)
        
        # Create node features
        node_features = []
        for element in elements:
            if element in self.element_features:
                node_features.append(self.element_features[element])
            else:
                # Default features for unknown elements
                node_features.append(np.zeros(77))
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Create edges based on distance cutoff
        edge_indices = []
        edge_features = []
        cutoff = 15.0  # Angstrom
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    edge_indices.extend([[i, j], [j, i]])  # Undirected graph
                    edge_features.extend([[dist], [dist]])
        
        if len(edge_indices) == 0:
            # Create self-loops if no edges found
            edge_indices = [[i, i] for i in range(num_atoms)]
            edge_features = [[0.0] for _ in range(num_atoms)]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic crystal data for demonstration"""
    data = []
    labels = []
    
    preprocessor = DataPreprocessor()
    ph_processor = PersistentHomologyProcessor()
    
    for i in range(num_samples):
        # Generate random crystal structure
        num_atoms = np.random.randint(5, 20)
        positions = np.random.rand(num_atoms, 3) * 10  # Random positions
        elements = np.random.choice(['H', 'C', 'N', 'O', 'Fe', 'Cu'], size=num_atoms)
        
        # Create graph
        graph_data = preprocessor.crystal_to_graph(positions, elements)
        
        # Create persistent homology features
        ph_features = ph_processor.extract_atom_specific_features(positions, elements)
        
        # Random labels (0: trivial, 1: topological)
        label = np.random.randint(0, 2)
        
        data.append((graph_data, ph_features))
        labels.append(label)
    
    return data, labels

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the topological classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_graphs, batch_ph, batch_labels in train_loader:
            batch_graphs = batch_graphs.to(device)
            batch_ph = batch_ph.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_graphs, batch_ph)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch_graphs, batch_ph, batch_labels in val_loader:
                batch_graphs = batch_graphs.to(device)
                batch_ph = batch_ph.to(device)
                
                outputs = model(batch_graphs, batch_ph)
                predictions = torch.argmax(outputs, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(batch_labels.numpy())
        
        val_accuracy = accuracy_score(val_true, val_predictions)
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.4f}')
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    confidence_scores = []
    
    with torch.no_grad():
        for batch_graphs, batch_ph, batch_labels in test_loader:
            batch_graphs = batch_graphs.to(device)
            batch_ph = batch_ph.to(device)
            
            outputs = model(batch_graphs, batch_ph)
            probabilities = F.softmax(outputs, dim=1)
            batch_predictions = torch.argmax(outputs, dim=1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            true_labels.extend(batch_labels.numpy())
            confidence_scores.extend(torch.max(probabilities, dim=1)[0].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print("Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Confidence: {np.mean(confidence_scores):.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions,
        'true_labels': true_labels,
        'confidence_scores': confidence_scores
    }

def create_data_loaders(data, labels, batch_size=32, test_size=0.2, val_size=0.1):
    """Create data loaders for training, validation, and testing"""
    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=val_size/(1-test_size), random_state=42, stratify=train_labels
    )
    
    # Custom collate function
    def collate_fn(batch):
        graphs, ph_features, labels = zip(*batch)
        
        # Batch graphs
        from torch_geometric.data import Batch
        batched_graphs = Batch.from_data_list(graphs)
        
        # Stack other features
        ph_tensor = torch.stack([torch.tensor(ph, dtype=torch.float32) for ph in ph_features])
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return batched_graphs, ph_tensor, label_tensor
    
    # Create datasets
    train_dataset = [(data[i][0], data[i][1], train_labels[i]) for i in range(len(train_data))]
    val_dataset = [(data[i][0], data[i][1], val_labels[i]) for i in range(len(val_data))]
    test_dataset = [(data[i][0], data[i][1], test_labels[i]) for i in range(len(test_data))]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

def main():
    """Main execution function"""
    print("Generating synthetic data...")
    data, labels = generate_synthetic_data(num_samples=1000)
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(data, labels, batch_size=16)
    
    print("Initializing model...")
    model = TopologicalClassifier(num_classes=2)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    print("Training model...")
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=20)
    
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    return model, results

if __name__ == "__main__":
    # Install required packages (run these in your environment):
    # pip install torch torch-geometric scikit-learn matplotlib ripser
    
    model, results = main()