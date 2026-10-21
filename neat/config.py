from dataclasses import dataclass


@dataclass(frozen=True)
class NeatConfig:
    num_inputs : int
    num_outputs: int
    pop_size: int = 150
    w_max: float = 5.0
    p_perturb: float = 0.9 
    sigma: float = 0.5
    activations: tuple[str, ...] = ("tanh", "sigmoid", "sin", "gauss", "relu", "abs")
    p_weight: float = 0.8
    p_add_conn: float = 0.1
    p_add_node: float = 0.05
    p_activation: float = 0.05
    p_toggle: float = 0.01
    distance_coefficients: tuple[float, float, float] = (0.5, 0.5, 0.3)
    n_small_cutoff: int = 20   # N=1 when both genomes have fewer conns than this
    compat_threshold: float = 3.0
    max_stagnation: int = 15
    survival_threshold: float = 0.3
    p_crossover: float = 0.75
    elitism_min_size: int = 5   # species bigger than this keep their best unchanged
    output_activation: str = "sigmoid"
    dynamic_threshold: bool = True      # nudge compat_threshold toward target_species
    target_species: tuple[int, int] = (8, 12)
    threshold_step: float = 0.1
    threshold_min: float = 0.3
    min_species_size: int = 2         # children guaranteed to every surviving species
