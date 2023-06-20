import neat
import visualize
import random
import os

# Configuring the network to be evolved
config_path = os.path.join(os.getcwd(), 'config-feedforward.txt')

# Define a simple evaluation function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi in range(-10,10):
            output = net.activate([xi])
            genome.fitness -= (output[0] - xi ** 2) ** 2

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for up to 30 generations.
winner = p.run(eval_genomes, 30)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi in range(-10,10):
    output = winner_net.activate([xi])
    print("input {!r}, expected output {!r}, got {!r}".format(
        xi, xi**2, output[0]
    ))

# Visualize the network
visualize.draw_net(config, winner, True)
