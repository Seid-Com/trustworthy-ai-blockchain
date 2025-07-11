import random
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import hashlib
import json

class AttackSimulator:
    def __init__(self):
        self.attacks_launched = 0
        self.attacks_detected = 0
        self.false_positives = 0
        self.attack_history = []
        self.detection_methods = ["blockchain_integrity", "consensus_validation", "anomaly_detection", "statistical_analysis"]
    
    def data_poisoning_attack(self, poison_percentage: float, num_malicious_clients: int) -> Dict[str, Any]:
        """Simulate a data poisoning attack"""
        start_time = time.time()
        
        # Calculate number of poisoned samples
        total_samples = random.randint(1000, 5000)
        poisoned_samples = int(total_samples * poison_percentage / 100)
        
        # Simulate attack execution
        attack_data = {
            "attack_type": "data_poisoning",
            "poison_percentage": poison_percentage,
            "malicious_clients": num_malicious_clients,
            "poisoned_samples": poisoned_samples,
            "total_samples": total_samples,
            "attack_vector": "label_flipping",
            "timestamp": datetime.now().isoformat()
        }
        
        # Simulate detection
        detection_time = random.uniform(0.5, 3.0)
        time.sleep(detection_time)
        
        # Detection probability increases with poison percentage
        detection_prob = min(0.95, 0.5 + (poison_percentage / 100))
        detected = random.random() < detection_prob
        
        if detected:
            self.attacks_detected += 1
        
        # Simulate mitigation
        mitigation_applied = detected and random.random() > 0.1
        
        attack_result = {
            "attack_active": True,
            "poisoned_samples": poisoned_samples,
            "detection_time": detection_time,
            "detected": detected,
            "mitigated": mitigation_applied,
            "detection_method": random.choice(self.detection_methods),
            "impact_severity": self.calculate_impact_severity(poison_percentage, detected, mitigation_applied)
        }
        
        # Record attack
        self.attacks_launched += 1
        self.attack_history.append({
            "attack_data": attack_data,
            "result": attack_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return attack_result
    
    def model_poisoning_attack(self, gradient_scale: float, target_accuracy_drop: float) -> Dict[str, Any]:
        """Simulate a model poisoning attack"""
        start_time = time.time()
        
        # Simulate malicious gradient manipulation
        manipulated_params = random.randint(10, 50)
        
        # Calculate actual accuracy drop based on gradient scale
        actual_accuracy_drop = min(target_accuracy_drop, gradient_scale * 5 + np.random.normal(0, 2))
        
        # Simulate detection
        detection_time = random.uniform(0.2, 1.5)
        time.sleep(detection_time)
        
        # Detection probability increases with gradient scale
        detection_prob = min(0.98, 0.6 + (gradient_scale / 20))
        detected = random.random() < detection_prob
        
        if detected:
            self.attacks_detected += 1
        
        attack_result = {
            "attack_active": True,
            "manipulated_params": manipulated_params,
            "gradient_scale": gradient_scale,
            "accuracy_drop": actual_accuracy_drop,
            "detection_time": detection_time,
            "detected": detected,
            "detection_method": "consensus_validation" if detected else "undetected"
        }
        
        # Record attack
        self.attacks_launched += 1
        self.attack_history.append({
            "attack_type": "model_poisoning",
            "result": attack_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return attack_result
    
    def sybil_attack(self, fake_identities: int, coordination_level: float) -> Dict[str, Any]:
        """Simulate a Sybil attack"""
        start_time = time.time()
        
        # Simulate fake identity creation
        fake_clients = []
        for i in range(fake_identities):
            fake_client = {
                "client_id": f"fake_client_{i}",
                "reputation": random.uniform(0.3, 0.7),
                "creation_time": datetime.now().isoformat(),
                "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            }
            fake_clients.append(fake_client)
        
        # Calculate consensus disruption
        consensus_disruption = min(80, fake_identities * coordination_level * 10)
        
        # Simulate detection
        detection_time = random.uniform(1.0, 5.0)
        time.sleep(detection_time)
        
        # Detection based on identity analysis
        detection_rate = min(95, 50 + fake_identities * 2)
        detected_identities = int(fake_identities * detection_rate / 100)
        
        if detected_identities > 0:
            self.attacks_detected += 1
        
        # Simulate mitigation
        mitigation_applied = detected_identities >= fake_identities * 0.7
        
        attack_result = {
            "fake_identities": fake_identities,
            "coordination_level": coordination_level,
            "consensus_disruption": consensus_disruption,
            "detection_rate": detection_rate,
            "detected_identities": detected_identities,
            "mitigated": mitigation_applied,
            "detection_method": "identity_verification"
        }
        
        # Record attack
        self.attacks_launched += 1
        self.attack_history.append({
            "attack_type": "sybil_attack",
            "result": attack_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return attack_result
    
    def inference_attack(self, target_samples: int, inference_method: str) -> Dict[str, Any]:
        """Simulate an inference attack"""
        start_time = time.time()
        
        # Simulate information leakage based on method
        if inference_method == "Gradient Inversion":
            base_leakage = 0.15
            method_multiplier = 1.5
        else:  # Model Inversion
            base_leakage = 0.10
            method_multiplier = 1.2
        
        information_leaked = min(50, base_leakage * target_samples * method_multiplier)
        
        # Calculate privacy loss
        privacy_loss = information_leaked / 100
        
        # Simulate detection
        detection_time = random.uniform(0.8, 3.0)
        time.sleep(detection_time)
        
        # Detection probability
        detection_prob = 0.7 if information_leaked > 20 else 0.4
        detected = random.random() < detection_prob
        
        if detected:
            self.attacks_detected += 1
        
        # Simulate countermeasures
        countermeasures = []
        if detected:
            countermeasures = random.sample(
                ["differential_privacy", "gradient_clipping", "secure_aggregation", "noise_injection"],
                random.randint(1, 3)
            )
        
        attack_result = {
            "target_samples": target_samples,
            "inference_method": inference_method,
            "information_leaked": information_leaked,
            "privacy_loss": privacy_loss,
            "detection_time": detection_time,
            "detected": detected,
            "countermeasures": countermeasures
        }
        
        # Record attack
        self.attacks_launched += 1
        self.attack_history.append({
            "attack_type": "inference_attack",
            "result": attack_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return attack_result
    
    def adversarial_attack(self, perturbation_strength: float, target_model: str) -> Dict[str, Any]:
        """Simulate an adversarial attack on model predictions"""
        start_time = time.time()
        
        # Simulate adversarial example generation
        num_adversarial_examples = random.randint(50, 200)
        success_rate = min(95, 30 + perturbation_strength * 20)
        
        # Calculate attack metrics
        fooled_predictions = int(num_adversarial_examples * success_rate / 100)
        
        # Simulate detection
        detection_time = random.uniform(0.1, 0.5)
        time.sleep(detection_time)
        
        # Detection based on perturbation strength
        detection_prob = min(0.9, 0.3 + perturbation_strength / 10)
        detected = random.random() < detection_prob
        
        if detected:
            self.attacks_detected += 1
        
        attack_result = {
            "perturbation_strength": perturbation_strength,
            "num_adversarial_examples": num_adversarial_examples,
            "success_rate": success_rate,
            "fooled_predictions": fooled_predictions,
            "detection_time": detection_time,
            "detected": detected,
            "defense_mechanism": "adversarial_training" if detected else "none"
        }
        
        # Record attack
        self.attacks_launched += 1
        self.attack_history.append({
            "attack_type": "adversarial_attack",
            "result": attack_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return attack_result
    
    def calculate_impact_severity(self, attack_strength: float, detected: bool, mitigated: bool) -> str:
        """Calculate the severity of attack impact"""
        if mitigated:
            return "Low"
        elif detected:
            return "Medium" if attack_strength > 30 else "Low"
        else:
            if attack_strength > 50:
                return "High"
            elif attack_strength > 25:
                return "Medium"
            else:
                return "Low"
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get attack detection statistics"""
        detection_rate = (self.attacks_detected / max(1, self.attacks_launched)) * 100
        
        # Calculate average response time
        response_times = []
        for attack in self.attack_history:
            if "detection_time" in attack.get("result", {}):
                response_times.append(attack["result"]["detection_time"])
        
        avg_response_time = np.mean(response_times) if response_times else 0
        
        return {
            "attacks_launched": self.attacks_launched,
            "attacks_detected": self.attacks_detected,
            "false_positives": self.false_positives,
            "detection_rate": detection_rate,
            "avg_response_time": avg_response_time,
            "total_incidents": len(self.attack_history)
        }
    
    def generate_attack_report(self) -> Dict[str, Any]:
        """Generate comprehensive attack report"""
        attack_types = {}
        severity_distribution = {"Low": 0, "Medium": 0, "High": 0}
        
        for attack in self.attack_history:
            attack_type = attack.get("attack_type", "unknown")
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            
            severity = attack.get("result", {}).get("impact_severity", "Low")
            severity_distribution[severity] += 1
        
        return {
            "total_attacks": len(self.attack_history),
            "attack_types": attack_types,
            "severity_distribution": severity_distribution,
            "detection_statistics": self.get_detection_statistics(),
            "most_common_attack": max(attack_types.items(), key=lambda x: x[1])[0] if attack_types else "None",
            "report_timestamp": datetime.now().isoformat()
        }
    
    def simulate_defense_mechanism(self, defense_type: str) -> Dict[str, Any]:
        """Simulate various defense mechanisms"""
        defense_mechanisms = {
            "byzantine_fault_tolerance": {
                "effectiveness": 0.85,
                "overhead": 0.15,
                "description": "PBFT consensus mechanism"
            },
            "anomaly_detection": {
                "effectiveness": 0.75,
                "overhead": 0.08,
                "description": "Statistical anomaly detection"
            },
            "secure_aggregation": {
                "effectiveness": 0.80,
                "overhead": 0.12,
                "description": "Cryptographic secure aggregation"
            },
            "differential_privacy": {
                "effectiveness": 0.70,
                "overhead": 0.05,
                "description": "Noise injection for privacy"
            }
        }
        
        defense = defense_mechanisms.get(defense_type, {
            "effectiveness": 0.5,
            "overhead": 0.1,
            "description": "Unknown defense mechanism"
        })
        
        # Add some randomness
        defense["effectiveness"] += np.random.normal(0, 0.05)
        defense["overhead"] += np.random.normal(0, 0.02)
        
        return defense
    
    def get_attack_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of attacks"""
        timeline = []
        for attack in self.attack_history:
            timeline.append({
                "timestamp": attack["timestamp"],
                "attack_type": attack.get("attack_type", "unknown"),
                "detected": attack.get("result", {}).get("detected", False),
                "severity": attack.get("result", {}).get("impact_severity", "Low")
            })
        
        return sorted(timeline, key=lambda x: x["timestamp"])
    
    def reset_statistics(self):
        """Reset attack statistics"""
        self.attacks_launched = 0
        self.attacks_detected = 0
        self.false_positives = 0
        self.attack_history = []
