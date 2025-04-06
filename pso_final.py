import numpy as np

C_1 = 0.9 # Cognitive Component
C_2 = 1 # Social Component
MAX_ITER = 300
NO_OF_PARTICLES = 10

MAX_BOUND_X = 10
MIN_BOUND_X = -10

MAX_BOUND_Y = 9
MIN_BOUND_Y = -12

X = 0
Y = 1

# objective function
def f(x, y):
    return (x - 4.0) ** 2 + (y - 7.0) ** 2

# Two variable PSO
def optimize(f):
    xs = (MAX_BOUND_X - MIN_BOUND_X) * np.random.random_sample(size=(NO_OF_PARTICLES, 1)) + MIN_BOUND_X
    ys = (MAX_BOUND_Y - MIN_BOUND_Y) * np.random.random_sample(size=(NO_OF_PARTICLES, 1)) + MIN_BOUND_Y

    # positions of each particle
    positions = np.hstack([xs, ys])

    # personal bests of each particle
    personal_bests = np.zeros((NO_OF_PARTICLES,))
    for idx in range(NO_OF_PARTICLES):
        personal_bests[idx] = f(positions[idx][X], positions[idx][Y])

    # the position associated with the personal best of each particle
    personal_best_positions = positions.copy()

    # global best among all particles
    global_best = np.min(personal_bests)
    global_best_position = positions[np.argmin(personal_bests)]

    # initialize velocities of each particle
    velocities = np.zeros_like(positions)

    for iters in range(MAX_ITER):
        if global_best == 0:
            break

        r_1 = np.random.random_sample(size=positions.shape)
        r_2 = np.random.random_sample(size=positions.shape)

        # inertia that is a slowly decreasing function
        w = 1/(iters + 1) + 0.3

        # update each particle's velocity and position
        velocities = w * velocities + C_1 * r_1 * (personal_best_positions - positions) + C_2 * r_2 * (global_best_position - positions)
        positions += velocities

        # clip the particle's positions so that they stay inside the given bounds
        positions[:, X] = np.clip(positions[:, X], MIN_BOUND_X, MAX_BOUND_X)
        # alternatively, positions[:, positions[:, X] > MAX_BOUND_X] = MAX_BOUND_X and the same for the MIN_BOUND_X case can be done
        positions[:, Y] = np.clip(positions[:, Y], MIN_BOUND_Y, MAX_BOUND_Y)

        # calculate the objective of each particle at it's current position
        f_values = np.zeros_like(personal_bests)
        for idx in range(NO_OF_PARTICLES):
            f_values[idx] = f(positions[idx][X], positions[idx][Y])

        # # if the new objective of a given particle is lesser than it's personal best
        # # then the personal best is updated. the particle's personal best position is
        # # also updated

        condition = f_values < personal_bests
        # read the condition as "where f_values are better" or "where personal_bests are worse"
        personal_bests[condition] = f_values[condition]
        # Get positions where f_values are better and assign it to personal_best_positions
        # where personal_bests are worse
        personal_best_positions[condition] = positions[condition].copy()

        # ALTERNATIVELY:
        # for idx in range(NO_OF_PARTICLES):
        #     if f_values[idx] < personal_bests[idx]:
        #         personal_bests[idx] = f_values[idx]
        #         personal_best_positions[idx] = positions[idx].copy()

        global_best = np.min(personal_bests)
        global_best_position = personal_best_positions[np.argmin(personal_bests)]
        print(f"Function evaluated at {global_best_position} gives {global_best}")

optimize(f)