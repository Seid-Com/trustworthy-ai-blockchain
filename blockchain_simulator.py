import hashlib
import json
import time
from datetime import datetime
import random
import numpy as np
from typing import Dict, List, Any, Optional

class Block:
    def __init__(self, index: int, timestamp: str, data: List[Dict], previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.merkle_root = self.calculate_merkle_root()
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate the Merkle root of the data"""
        if not self.data:
            return hashlib.sha256(b"").hexdigest()
        
        # Create hashes of all data entries
        hashes = []
        for entry in self.data:
            entry_string = json.dumps(entry, sort_keys=True)
            hashes.append(hashlib.sha256(entry_string.encode()).hexdigest())
        
        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last hash if odd number
            
            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level
        
        return hashes[0] if hashes else hashlib.sha256(b"").hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "hash": self.hash
        }

class BlockchainSimulator:
    def __init__(self):
        self.chain = []
        self.pending_data = []
        self.mining_difficulty = 2
        self.mining_reward = 10
        self.data_entries = 0
        self.model_updates = 0
        self.integrity_violations = 0
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=datetime.now().isoformat(),
            data=[{"type": "genesis", "message": "Genesis Block"}],
            previous_hash="0"
        )
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_data_entry(self, data: Dict[str, Any]) -> str:
        """Add a data entry to the blockchain"""
        # Create data hash for integrity
        data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        # Add metadata
        data_entry = {
            "type": "data_entry",
            "data": data,
            "data_hash": data_hash,
            "timestamp": datetime.now().isoformat(),
            "entry_id": f"data_{self.data_entries}"
        }
        
        # Add to pending data
        self.pending_data.append(data_entry)
        self.data_entries += 1
        
        # Mine block if enough pending data
        if len(self.pending_data) >= 3:
            self.mine_pending_block()
        
        return data_hash
    
    def add_model_update(self, client_id: str, update: Dict[str, Any]) -> str:
        """Add a model update to the blockchain"""
        # Create update hash
        update_hash = hashlib.sha256(json.dumps(update, sort_keys=True).encode()).hexdigest()
        
        # Add metadata
        update_entry = {
            "type": "model_update",
            "client_id": client_id,
            "update": update,
            "update_hash": update_hash,
            "timestamp": datetime.now().isoformat(),
            "update_id": f"update_{self.model_updates}"
        }
        
        # Add to pending data
        self.pending_data.append(update_entry)
        self.model_updates += 1
        
        # Mine block if enough pending data
        if len(self.pending_data) >= 2:
            self.mine_pending_block()
        
        return update_hash
    
    def mine_pending_block(self):
        """Mine a new block with pending data"""
        if not self.pending_data:
            return
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            data=self.pending_data.copy(),
            previous_hash=self.get_latest_block().hash
        )
        
        # Proof of Work simulation (simplified)
        while not new_block.hash.startswith("0" * self.mining_difficulty):
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()
        
        # Add block to chain
        self.chain.append(new_block)
        
        # Clear pending data
        self.pending_data = []
    
    def validate_chain(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is valid
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if current block points to previous block
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_blocks = len(self.chain)
        integrity_score = 100.0 if self.validate_chain() else 0.0
        
        # Calculate integrity score with some randomness for demonstration
        if integrity_score == 100.0:
            integrity_score = max(95.0, 100.0 - random.uniform(0, 2))
        
        return {
            "total_blocks": total_blocks,
            "data_entries": self.data_entries,
            "model_updates": self.model_updates,
            "integrity_score": integrity_score,
            "chain_valid": self.validate_chain(),
            "pending_transactions": len(self.pending_data)
        }
    
    def get_recent_blocks(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent blocks from the chain"""
        recent_blocks = self.chain[-count:] if len(self.chain) >= count else self.chain
        return [block.to_dict() for block in reversed(recent_blocks)]
    
    def analyze_data_integrity(self) -> Dict[str, Any]:
        """Analyze data integrity across the blockchain"""
        integrity_entries = []
        
        for block in self.chain:
            for entry in block.data:
                if entry.get("type") == "data_entry":
                    # Simulate integrity check
                    integrity_score = random.uniform(0.90, 1.0)
                    
                    # Add some anomalies for demonstration
                    if random.random() < 0.05:  # 5% chance of anomaly
                        integrity_score = random.uniform(0.70, 0.89)
                        self.integrity_violations += 1
                    
                    integrity_entries.append({
                        "timestamp": entry["timestamp"],
                        "integrity_score": integrity_score,
                        "entry_id": entry.get("entry_id", "unknown")
                    })
        
        return {
            "entries": integrity_entries,
            "total_violations": self.integrity_violations,
            "average_integrity": np.mean([e["integrity_score"] for e in integrity_entries]) if integrity_entries else 1.0
        }
    
    def get_activity_data(self) -> Dict[str, List]:
        """Get blockchain activity data for visualization"""
        timestamps = []
        transactions = []
        
        for block in self.chain:
            timestamps.append(block.timestamp)
            transactions.append(len(block.data))
        
        return {
            "timestamps": timestamps,
            "transactions": transactions
        }
    
    def get_data_provenance(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """Get the provenance of a specific data entry"""
        for block in self.chain:
            for entry in block.data:
                if entry.get("data_hash") == data_hash:
                    return {
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "timestamp": entry["timestamp"],
                        "data": entry["data"],
                        "provenance_verified": True
                    }
        return None
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in the blockchain"""
        anomalies = []
        
        for block in self.chain:
            for entry in block.data:
                # Simulate anomaly detection
                if random.random() < 0.02:  # 2% chance of anomaly
                    anomalies.append({
                        "block_index": block.index,
                        "entry_id": entry.get("entry_id", "unknown"),
                        "anomaly_type": random.choice(["data_corruption", "timestamp_mismatch", "hash_inconsistency"]),
                        "severity": random.choice(["low", "medium", "high"]),
                        "timestamp": entry["timestamp"]
                    })
        
        return anomalies
