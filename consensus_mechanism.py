import random
import time
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class PBFTValidator:
    def __init__(self, validator_id: str, is_byzantine: bool = False):
        self.validator_id = validator_id
        self.is_byzantine = is_byzantine
        self.reputation_score = 1.0
        self.view_number = 0
        self.sequence_number = 0
        self.message_log = []
        self.consensus_history = []
        self.last_activity = datetime.now()
    
    def validate_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Validate a consensus proposal"""
        if self.is_byzantine:
            # Byzantine nodes randomly reject or approve
            return random.random() > 0.3
        
        # Normal validation logic
        try:
            # Check if proposal has required fields
            required_fields = ["client_id", "update_hash", "timestamp"]
            if not all(field in proposal for field in required_fields):
                return False
            
            # Check if timestamp is reasonable
            proposal_time = datetime.fromisoformat(proposal["timestamp"])
            time_diff = abs((datetime.now() - proposal_time).total_seconds())
            if time_diff > 300:  # 5 minutes threshold
                return False
            
            # Check if update hash is valid
            if len(proposal["update_hash"]) != 64:  # SHA-256 length
                return False
            
            return True
        except:
            return False
    
    def create_vote(self, proposal: Dict[str, Any], vote_type: str) -> Dict[str, Any]:
        """Create a vote for a proposal"""
        vote = {
            "validator_id": self.validator_id,
            "proposal_hash": hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest(),
            "vote_type": vote_type,  # "pre-prepare", "prepare", "commit"
            "view_number": self.view_number,
            "sequence_number": self.sequence_number,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add vote to message log
        self.message_log.append(vote)
        self.last_activity = datetime.now()
        
        return vote
    
    def get_status(self) -> Dict[str, Any]:
        """Get validator status"""
        return {
            "validator_id": self.validator_id,
            "is_byzantine": self.is_byzantine,
            "reputation_score": self.reputation_score,
            "view_number": self.view_number,
            "sequence_number": self.sequence_number,
            "message_count": len(self.message_log),
            "last_activity": self.last_activity.isoformat(),
            "consensus_participated": len(self.consensus_history)
        }

class PBFTConsensus:
    def __init__(self, num_validators: int = 10):
        self.num_validators = num_validators
        self.validators = {}
        self.byzantine_count = 0
        self.consensus_rounds = 0
        self.successful_rounds = 0
        self.consensus_history = []
        self.current_view = 0
        self.sequence_number = 0
        
        # Initialize validators
        self.initialize_validators()
    
    def initialize_validators(self):
        """Initialize PBFT validators"""
        self.validators = {}
        for i in range(self.num_validators):
            validator_id = f"validator_{i+1}"
            is_byzantine = i < self.byzantine_count
            self.validators[validator_id] = PBFTValidator(validator_id, is_byzantine)
    
    def set_validators(self, num_validators: int):
        """Set the number of validators"""
        if num_validators != self.num_validators:
            self.num_validators = num_validators
            self.initialize_validators()
    
    def set_byzantine_nodes(self, byzantine_count: int):
        """Set the number of Byzantine nodes"""
        max_byzantine = (self.num_validators - 1) // 3
        self.byzantine_count = min(byzantine_count, max_byzantine)
        
        # Reinitialize validators with new Byzantine count
        self.initialize_validators()
    
    def run_consensus(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Run PBFT consensus on a proposal"""
        start_time = time.time()
        
        # Phase 1: Pre-prepare
        primary_validator = self.get_primary_validator()
        pre_prepare_votes = self.phase_pre_prepare(proposal, primary_validator)
        
        # Phase 2: Prepare
        prepare_votes = self.phase_prepare(proposal, pre_prepare_votes)
        
        # Phase 3: Commit
        commit_votes = self.phase_commit(proposal, prepare_votes)
        
        # Determine consensus result
        consensus_time = time.time() - start_time
        
        # Calculate vote counts
        votes_for = sum(1 for vote in commit_votes if vote["vote_type"] == "commit")
        votes_against = len(commit_votes) - votes_for
        
        # PBFT requires 2f+1 votes for consensus (f = byzantine_count)
        required_votes = 2 * self.byzantine_count + 1
        consensus_reached = votes_for >= required_votes
        
        # Update statistics
        self.consensus_rounds += 1
        if consensus_reached:
            self.successful_rounds += 1
        
        # Record consensus result
        consensus_result = {
            "proposal": proposal,
            "consensus_reached": consensus_reached,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "consensus_time": consensus_time,
            "view_number": self.current_view,
            "sequence_number": self.sequence_number,
            "timestamp": datetime.now().isoformat(),
            "primary_validator": primary_validator.validator_id,
            "byzantine_tolerance": self.byzantine_count
        }
        
        self.consensus_history.append(consensus_result)
        self.sequence_number += 1
        
        return {
            "accepted": consensus_reached,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "consensus_time": consensus_time,
            "required_votes": required_votes,
            "participating_validators": len(commit_votes)
        }
    
    def get_primary_validator(self) -> PBFTValidator:
        """Get the primary validator for the current view"""
        validator_list = list(self.validators.values())
        primary_index = self.current_view % len(validator_list)
        return validator_list[primary_index]
    
    def phase_pre_prepare(self, proposal: Dict[str, Any], primary: PBFTValidator) -> List[Dict[str, Any]]:
        """Phase 1: Pre-prepare phase"""
        votes = []
        
        # Primary validator creates pre-prepare message
        if primary.validate_proposal(proposal):
            pre_prepare_vote = primary.create_vote(proposal, "pre-prepare")
            votes.append(pre_prepare_vote)
        
        return votes
    
    def phase_prepare(self, proposal: Dict[str, Any], pre_prepare_votes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 2: Prepare phase"""
        votes = []
        
        # All validators (except primary) vote on prepare
        for validator in self.validators.values():
            # Skip primary validator
            if validator.validator_id == self.get_primary_validator().validator_id:
                continue
            
            if validator.validate_proposal(proposal) and pre_prepare_votes:
                prepare_vote = validator.create_vote(proposal, "prepare")
                votes.append(prepare_vote)
        
        return votes
    
    def phase_commit(self, proposal: Dict[str, Any], prepare_votes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 3: Commit phase"""
        votes = []
        
        # All validators vote on commit if prepare phase succeeded
        required_prepare_votes = 2 * self.byzantine_count
        
        if len(prepare_votes) >= required_prepare_votes:
            for validator in self.validators.values():
                if validator.validate_proposal(proposal):
                    commit_vote = validator.create_vote(proposal, "commit")
                    votes.append(commit_vote)
        
        return votes
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get the validator network topology"""
        validators_info = []
        
        for validator in self.validators.values():
            status = validator.get_status()
            validators_info.append({
                "id": validator.validator_id,
                "status": "byzantine" if validator.is_byzantine else "honest",
                "reputation": validator.reputation_score
            })
        
        return {
            "validators": validators_info,
            "total_validators": self.num_validators,
            "byzantine_count": self.byzantine_count,
            "fault_tolerance": self.num_validators - 3 * self.byzantine_count
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        average_time = 0
        if self.consensus_history:
            average_time = np.mean([round_data["consensus_time"] for round_data in self.consensus_history])
        
        byzantine_tolerance = 100 * (self.num_validators - 3 * self.byzantine_count) / self.num_validators
        
        return {
            "total_rounds": self.consensus_rounds,
            "successful_rounds": self.successful_rounds,
            "success_rate": (self.successful_rounds / max(1, self.consensus_rounds)) * 100,
            "average_time": average_time,
            "byzantine_tolerance": byzantine_tolerance,
            "current_view": self.current_view,
            "sequence_number": self.sequence_number
        }
    
    def validate_model_updates(self) -> Dict[str, Dict[str, Any]]:
        """Validate model updates from clients"""
        validation_results = {}
        
        # Simulate validation of recent model updates
        for i in range(min(5, len(self.validators))):
            client_id = f"client_{i+1}"
            
            # Create a mock model update
            model_update = {
                "client_id": client_id,
                "update_hash": hashlib.sha256(f"update_{i}_{time.time()}".encode()).hexdigest(),
                "timestamp": datetime.now().isoformat(),
                "parameters": {"layer_1": random.random(), "layer_2": random.random()}
            }
            
            # Run consensus on the update
            consensus_result = self.run_consensus(model_update)
            
            validation_results[client_id] = {
                "valid": consensus_result["accepted"],
                "confidence": min(1.0, consensus_result["votes_for"] / max(1, consensus_result["votes_for"] + consensus_result["votes_against"])),
                "votes_for": consensus_result["votes_for"],
                "votes_against": consensus_result["votes_against"]
            }
        
        return validation_results
    
    def verify_identities(self) -> Dict[str, Dict[str, Any]]:
        """Verify client identities to detect Sybil attacks"""
        identity_results = {}
        
        # Simulate identity verification
        for i in range(15):  # Check more clients than validators
            client_id = f"client_{i+1}"
            
            # Simulate identity verification logic
            is_legitimate = random.random() > 0.2  # 80% legitimate
            reputation_score = random.uniform(0.5, 1.0) if is_legitimate else random.uniform(0.0, 0.4)
            
            identity_results[client_id] = {
                "legitimate": is_legitimate,
                "reputation": reputation_score,
                "verification_time": datetime.now().isoformat()
            }
        
        return identity_results
    
    def handle_view_change(self):
        """Handle view change in case of primary failure"""
        self.current_view += 1
        
        # Reset validator states
        for validator in self.validators.values():
            validator.view_number = self.current_view
            validator.message_log = []
    
    def get_consensus_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent consensus log entries"""
        return self.consensus_history[-limit:] if len(self.consensus_history) >= limit else self.consensus_history
