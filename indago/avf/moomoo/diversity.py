import numpy as np
import autograd.numpy as anp
from scipy.spatial.distance import euclidean
from indago.avf.env_configuration import EnvConfiguration
from math import sqrt

# Original Parking Diversity calculation
# Uses different diversity metrics for each part of the environment
# Rather than euclidean distance for the entire environment.
class ParkingDiversity():
    def __init__(self):
        pass
      
    def weighted_jaccard_distance(self, set1, set2, weights):
            """
            Compute the weighted Jaccard distance between two sets of parked vehicles.
            """
            set1, set2 = set(set1), set(set2)
            intersection = set1 & set2
            union = set1 | set2
            weighted_intersection = sum(weights[i] for i in intersection)
            weighted_union = sum(weights[i] for i in union)
            if weighted_union == 0:  # Avoid division by zero
                return 0.0
            return 1 - (weighted_intersection / weighted_union)

    def euclidean_distance(self, pos1, pos2, scale_x=20, scale_y=10):
            """
            Compute the normalized Euclidean distance between two positions.
            """
            dx = (pos1[0] - pos2[0]) / scale_x
            dy = (pos1[1] - pos2[1]) / scale_y
            return sqrt(dx**2 + dy**2)

    def environment_diversity(self, env1, env2, weights, parked_weights):
            """
            Compute the diversity metric between two environments.
            """
            # goal_lane1, heading1, parked1, pos1 = env1
            # goal_lane2, heading2, parked2, pos2 = env2
            
            # Individual distance components
            d_goal_lane = abs(env1.goal_lane_idx - env2.goal_lane_idx) / 19
            d_heading = abs(env1.heading_ego - env2.heading_ego)
            d_parked = self.weighted_jaccard_distance(env1.parked_vehicles_lane_indices, env2.parked_vehicles_lane_indices, parked_weights)
            d_position = self.euclidean_distance(env1.position_ego, env2.position_ego)
            
            # Combine distances with weights
            diversity = (weights['goal_lane'] * d_goal_lane +
                        weights['heading'] * d_heading +
                        weights['parked'] * d_parked +
                        weights['position'] * d_position)
            return diversity

    # Weights for the diversity metric
    weights = {
        'goal_lane': 0.25,
        'heading': 0.25,
        'parked': 0.25,
        'position': 0.25
    }

    # Weights for the parked vehicles
    parked_weights = {i: 1 / (i + 1) for i in range(20)}  # Example: Decreasing weights by index

    def evaluate_child(self, fitness_fn, saved_environments, weights, parked_weights, child):
        chromosome = child[0]
        if chromosome.fitness is None:
            chromosome.compute_fitness(fitness_fn=fitness_fn)
        fitness = chromosome.fitness

        if saved_environments:
            # Compute diversity against all saved environments
            individual_diversity = [
                self.environment_diversity(chromosome.env_config, saved_env, weights, parked_weights)
                for saved_env in saved_environments
            ]
            # Aggregate diversity scores (e.g., take the mean)
            aggregated_diversity = min(individual_diversity)
            diversity_score = abs(1 - aggregated_diversity)
        else:
            diversity_score = 0

        return fitness, diversity_score

# Simple Parking Diversity
# Uses the euclidean distance on the environment
class ParkingDiversitySimple():

    def evaluate_child(self, fitness_fn, dataset, saved_environments, child):
        chromosome = child[0]
        if chromosome.fitness is None:
            chromosome.compute_fitness(fitness_fn=fitness_fn)
        # Chromosome fitness is already (1 - fitness) in the fitness_fn for minimization

        if saved_environments:
            # Compute diversity against all saved environments
            individual_diversity = [
                dataset.compute_distance(chromosome.env_config, saved_env)
                for saved_env in saved_environments
            ]
            # Aggregate diversity scores (e.g., take the mean)
            aggregated_diversity = np.mean(individual_diversity)
            diversity_score = abs(20 - aggregated_diversity) # Using a value of 20 here as the best diversity value was close to this
        else:
            diversity_score = 0

        # print(f"{chromosome.env_config.get_str()} | {fitness} | {diversity_score}")
        return chromosome.fitness, diversity_score


class DonkeyDiversity():
    def __init__(self):
        pass

    def parse_environment(self, env_str):
        """
        Parses the environment string and converts it into a list of 2D points.
        """
        instructions = env_str.split('@')
        x, y = 0, 0  # Start at origin
        points = [(x, y)]
        direction = 0  # 0: Right, 90: Up, 180: Left, 270: Down
        
        for instr in instructions:
            parts = instr.split()
            command = parts[0]
            value = float(parts[1])
            
            if command == 'S':  # Straight road
                dx, dy = np.cos(np.radians(direction)) * value, np.sin(np.radians(direction)) * value
                x, y = x + dx, y + dy
            elif command == 'L':  # Left turn (change direction counterclockwise)
                direction += value
            elif command == 'R':  # Right turn (change direction clockwise)
                direction -= value
            elif command == 'DY':  # Elevation change, ignored for 2D
                continue
            
            points.append((x, y))
    
        return np.array(points)

    def euclidean_distance(self, env1, env2):
        """
        Computes pairwise Euclidean distance between two environments.
        """
        points1 = self.parse_environment(env1)
        points2 = self.parse_environment(env2)
        
        if len(points1) != len(points2):
            raise ValueError("Environments must have the same number of points")
        
        distances = np.linalg.norm(points1 - points2, axis=1)
        return np.sum(distances)  # Total distance as a diversity measure

    def evaluate_child(self, fitness_fn, saved_environments, child):
        chromosome = child[0]
        if chromosome.fitness is None:
            chromosome.compute_fitness(fitness_fn=fitness_fn)
        fitness = chromosome.fitness

        if saved_environments:
            # Compute diversity against all saved environments
            individual_diversity = [
                self.euclidean_distance(chromosome.env_config, saved_env)
                for saved_env in saved_environments
            ]
            # Aggregate diversity scores (e.g., take the mean)
            aggregated_diversity = min(individual_diversity)
            diversity_score = abs(1 - aggregated_diversity)
        else:
            diversity_score = 0

        return fitness, diversity_score

# Simple Humanoid Diversity
# Uses the euclidean distance on the environment
class DonkeyDiversitySimple():

    def evaluate_child(self, fitness_fn, dataset, saved_environments, child):
        chromosome = child[0]
        if chromosome.fitness is None:
            chromosome.compute_fitness(fitness_fn=fitness_fn)
        # Chromosome fitness is already (1 - fitness) in the fitness_fn for minimization

        if saved_environments:
            # Compute diversity against all saved environments
            individual_diversity = [
                dataset.compute_distance(chromosome.env_config, saved_env)
                for saved_env in saved_environments
            ]
            # Aggregate diversity scores (e.g., take the mean)
            aggregated_diversity = np.mean(individual_diversity)
            diversity_score = abs(20 - aggregated_diversity) # Using a value of 20 here as the best diversity value was close to this
        else:
            diversity_score = 0

        # print(f"{chromosome.env_config.get_str()} | {fitness} | {diversity_score}")
        return chromosome.fitness, diversity_score
    
# Simple Humanoid Diversity
# Uses the euclidean distance on the environment
class HumanoidDiversitySimple():

    def evaluate_child(self, fitness_fn, dataset, saved_environments, child):
        chromosome = child[0]
        if chromosome.fitness is None:
            chromosome.compute_fitness(fitness_fn=fitness_fn)
        # Chromosome fitness is already (1 - fitness) in the fitness_fn for minimization

        if saved_environments:
            # Compute diversity against all saved environments
            individual_diversity = [
                dataset.compute_distance(chromosome.env_config, saved_env)
                for saved_env in saved_environments
            ]
            # Aggregate diversity scores (e.g., take the mean)
            aggregated_diversity = np.mean(individual_diversity)
            diversity_score = abs(20 - aggregated_diversity) # Using a value of 20 here as the best diversity value was close to this
        else:
            diversity_score = 0

        # print(f"{chromosome.env_config.get_str()} | {fitness} | {diversity_score}")
        return chromosome.fitness, diversity_score