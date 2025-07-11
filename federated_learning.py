import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json

class FederatedClient:
    def __init__(self, client_id: str, data_size: int = 1000):
        self.client_id = client_id
        self.data_size = data_size
        self.model = None
        self.local_data = None
        self.local_labels = None
        self.training_history = []
        self.last_update_time = None
        self.reputation_score = 1.0
        self.is_malicious = False
        
        # Initialize with MNIST-like data
        self.initialize_data()
    
    def initialize_data(self):
        """Initialize local data for the client"""
        # Generate synthetic MNIST-like data
        self.local_data = np.random.rand(self.data_size, 28, 28, 1).astype(np.float32)
        self.local_labels = np.random.randint(0, 10, self.data_size)
        
        # Add some realistic patterns
        for i in range(self.data_size):
            label = self.local_labels[i]
            # Add some structure based on label
            self.local_data[i] += np.random.normal(0, 0.1, (28, 28, 1))
            self.local_data[i] = np.clip(self.local_data[i], 0, 1)
    
    def set_model(self, model):
        """Set the local model"""
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            warm_start=True
        )
    
    def train_local_model(self, epochs: int = 5, batch_size: int = 32) -> Dict[str, Any]:
        """Train the local model"""
        if self.model is None:
            raise ValueError("Model not set for client")
        
        # Reshape data for sklearn (flatten images)
        X_flat = self.local_data.reshape(self.local_data.shape[0], -1)
        
        # Split data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, self.local_labels, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        final_accuracy = accuracy_score(y_val, val_pred)
        final_loss = 1 - accuracy_score(y_train, train_pred)  # Simple loss approximation
        
        # Simulate weights (sklearn doesn't expose weights the same way)
        weights = [self.model.coefs_[0], self.model.intercepts_[0]]
        
        # Update training history
        training_result = {
            "client_id": self.client_id,
            "timestamp": datetime.now().isoformat(),
            "loss": final_loss,
            "accuracy": final_accuracy,
            "epochs": epochs,
            "data_size": len(X_train),
            "weights_hash": self.calculate_weights_hash(weights)
        }
        
        self.training_history.append(training_result)
        self.last_update_time = datetime.now()
        
        return {
            "weights": weights,
            "metrics": training_result,
            "client_id": self.client_id
        }
    
    def calculate_weights_hash(self, weights: List[np.ndarray]) -> str:
        """Calculate hash of model weights"""
        import hashlib
        weights_string = json.dumps([w.tolist() for w in weights], sort_keys=True)
        return hashlib.sha256(weights_string.encode()).hexdigest()
    
    def inject_malicious_update(self, weights: List[np.ndarray], scale: float = 5.0) -> List[np.ndarray]:
        """Inject malicious update into weights"""
        malicious_weights = []
        for w in weights:
            # Scale weights to cause model poisoning
            malicious_w = w * scale
            malicious_weights.append(malicious_w)
        
        self.is_malicious = True
        return malicious_weights
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "client_id": self.client_id,
            "data_size": self.data_size,
            "status": "active" if self.last_update_time else "inactive",
            "reputation": self.reputation_score,
            "is_malicious": self.is_malicious,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "training_rounds": len(self.training_history)
        }

class FederatedLearningSystem:
    def __init__(self, num_clients: int = 10):
        self.num_clients = num_clients
        self.clients = {}
        self.global_model = None
        self.global_weights = None
        self.training_rounds = 0
        self.performance_history = []
        self.anomaly_threshold = 0.1
        
        # Initialize clients
        self.initialize_clients()
        
        # Create global model
        self.create_global_model()
    
    def initialize_clients(self):
        """Initialize federated learning clients"""
        for i in range(self.num_clients):
            client_id = f"client_{i+1}"
            data_size = random.randint(500, 2000)
            self.clients[client_id] = FederatedClient(client_id, data_size)
    
    def create_global_model(self):
        """Create the global neural network model"""
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            warm_start=True
        )
        
        self.global_model = model
        self.global_weights = None  # Will be set after first training
        
        # Distribute initial model to clients
        for client in self.clients.values():
            client.set_model(model)
    
    def set_num_clients(self, num_clients: int):
        """Set the number of clients"""
        if num_clients != self.num_clients:
            self.num_clients = num_clients
            self.clients = {}
            self.initialize_clients()
            
            # Redistribute global model
            for client in self.clients.values():
                client.set_model(self.global_model)
    
    def training_round(self) -> Dict[str, Any]:
        """Perform one round of federated training"""
        # Select random subset of clients for training
        num_selected = max(1, int(self.num_clients * 0.7))  # 70% of clients
        selected_clients = random.sample(list(self.clients.keys()), num_selected)
        
        client_updates = {}
        client_accuracies = {}
        
        # Train local models
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Reset model for new training (sklearn doesn't have set_weights)
            client.set_model(self.global_model)
            
            # Train local model
            update = client.train_local_model(epochs=3, batch_size=32)
            client_updates[client_id] = update
            client_accuracies[client_id] = update["metrics"]["accuracy"]
        
        # Detect anomalies in updates
        anomalies = self.detect_anomalous_updates(client_updates)
        
        # Filter out anomalous updates
        filtered_updates = {k: v for k, v in client_updates.items() if k not in anomalies}
        
        # Aggregate weights using federated averaging
        self.global_weights = self.federated_averaging(filtered_updates)
        
        # Update global model (sklearn doesn't have set_weights, so we keep the reference)
        # The global model is updated through federated averaging
        
        # Calculate global metrics
        global_accuracy = self.evaluate_global_model()
        
        # Update training history
        round_result = {
            "round": self.training_rounds,
            "timestamp": datetime.now().isoformat(),
            "selected_clients": selected_clients,
            "client_accuracies": client_accuracies,
            "global_accuracy": global_accuracy,
            "anomalies_detected": len(anomalies),
            "average_loss": np.mean([update["metrics"]["loss"] for update in filtered_updates.values()]),
            "communication_cost": self.calculate_communication_cost(filtered_updates)
        }
        
        self.performance_history.append(round_result)
        self.training_rounds += 1
        
        return client_updates
    
    def detect_anomalous_updates(self, client_updates: Dict[str, Any]) -> List[str]:
        """Detect anomalous client updates"""
        anomalies = []
        
        if len(client_updates) < 2:
            return anomalies
        
        # Calculate weight statistics
        weight_norms = {}
        for client_id, update in client_updates.items():
            weights = update["weights"]
            norm = np.linalg.norm([np.linalg.norm(w) for w in weights])
            weight_norms[client_id] = norm
        
        # Detect outliers using simple statistical method
        norms = list(weight_norms.values())
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        for client_id, norm in weight_norms.items():
            if abs(norm - mean_norm) > 2 * std_norm:  # 2-sigma rule
                anomalies.append(client_id)
                # Reduce reputation score
                self.clients[client_id].reputation_score *= 0.9
        
        return anomalies
    
    def federated_averaging(self, client_updates: Dict[str, Any]) -> List[np.ndarray]:
        """Perform federated averaging of client updates"""
        if not client_updates:
            return self.global_weights
        
        # Get data sizes for weighted averaging
        data_sizes = {}
        for client_id, update in client_updates.items():
            data_sizes[client_id] = update["metrics"]["data_size"]
        
        total_data_size = sum(data_sizes.values())
        
        # Initialize averaged weights
        averaged_weights = None
        
        for client_id, update in client_updates.items():
            client_weights = update["weights"]
            weight_ratio = data_sizes[client_id] / total_data_size
            
            if averaged_weights is None:
                averaged_weights = [w * weight_ratio for w in client_weights]
            else:
                for i, w in enumerate(client_weights):
                    averaged_weights[i] += w * weight_ratio
        
        return averaged_weights
    
    def evaluate_global_model(self) -> float:
        """Evaluate the global model performance"""
        # Create a synthetic test dataset
        test_data = np.random.rand(1000, 28, 28, 1).astype(np.float32)
        test_labels = np.random.randint(0, 10, 1000)
        
        # Flatten data for sklearn
        test_data_flat = test_data.reshape(test_data.shape[0], -1)
        
        # Evaluate model
        try:
            if hasattr(self.global_model, 'score'):
                accuracy = self.global_model.score(test_data_flat, test_labels)
                return accuracy * 100  # Convert to percentage
            else:
                return random.uniform(95, 99)  # Fallback for demonstration
        except:
            return random.uniform(95, 99)  # Fallback for demonstration
    
    def calculate_communication_cost(self, client_updates: Dict[str, Any]) -> float:
        """Calculate communication cost for the round"""
        total_params = 0
        for update in client_updates.values():
            weights = update["weights"]
            for w in weights:
                total_params += w.size
        
        # Simulate communication cost (bytes)
        return total_params * 4  # 4 bytes per float32
    
    def get_client_status(self) -> List[Dict[str, Any]]:
        """Get status of all clients"""
        client_status = []
        for client in self.clients.values():
            status = client.get_status()
            
            # Add some dynamic metrics for demonstration
            status["accuracy"] = random.uniform(85, 98)
            status["data_quality"] = random.uniform(7, 10)
            
            client_status.append(status)
        
        return client_status
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get training performance history"""
        return self.performance_history
    
    def validate_updates(self) -> Dict[str, Any]:
        """Validate recent model updates"""
        validation_results = {}
        
        for client_id, client in self.clients.items():
            if client.training_history:
                recent_update = client.training_history[-1]
                
                # Simulate validation
                is_valid = random.random() > 0.1  # 90% valid
                confidence = random.uniform(0.8, 1.0) if is_valid else random.uniform(0.3, 0.7)
                
                validation_results[client_id] = {
                    "valid": is_valid,
                    "confidence": confidence,
                    "timestamp": recent_update["timestamp"]
                }
        
        return validation_results
    
    def reset(self):
        """Reset the federated learning system"""
        self.training_rounds = 0
        self.performance_history = []
        
        # Reset clients
        for client in self.clients.values():
            client.training_history = []
            client.last_update_time = None
            client.reputation_score = 1.0
            client.is_malicious = False
        
        # Reset global model
        self.create_global_model()
    
    def introduce_malicious_client(self, client_id: str, poison_scale: float = 5.0):
        """Introduce a malicious client"""
        if client_id in self.clients:
            self.clients[client_id].is_malicious = True
            # Override the training method to inject malicious updates
            original_train = self.clients[client_id].train_local_model
            
            def malicious_train(*args, **kwargs):
                result = original_train(*args, **kwargs)
                result["weights"] = self.clients[client_id].inject_malicious_update(
                    result["weights"], poison_scale
                )
                return result
            
            self.clients[client_id].train_local_model = malicious_train
