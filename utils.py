import random
import numpy as np

def initialize_population(args):
    population = []
    for _ in range(args.pop_size):
        particle = [round(random.uniform(-10, 10),4) for _ in range(args.particle_length)]
        population.append(particle)
    return population


def evolve(population, gbest_individual, pbest_individuals, velocity_set, args):
    offspring = []
    new_velocity_set = []
    for i,particle in enumerate(population):
        new_particle, new_velocity = pso_evolve(particle, gbest_individual, pbest_individuals[i], velocity_set[i], args)
        offspring.append(new_particle)
        new_velocity_set.append(new_velocity)
    return offspring, new_velocity_set


def pso_evolve(particle, gbest, pbest, velocity, args):
    w_min, w_max = 0.3, 0.9
    c1_min, c1_max = 1.0, 2.0
    c2_min, c2_max = 1.0, 2.0

    w = w_max - (w_max-w_min)*args.curr_gen/args.num_generations
    c1 = c1_max - (c1_max-c1_min)*args.curr_gen/args.num_generations
    c2 = c2_min + (c2_max-c2_min)*args.curr_gen/args.num_generations

    cur_len = len(particle)

    # velocity calculation
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)
    new_velocity = np.asarray(velocity) * w + c1 * r1 * (np.asarray(pbest) - np.asarray(particle)) + c2 * r2 * (np.asarray(gbest) - np.asarray(particle))

    # boundary handling
    updated_velocity = []
    for i,vel in enumerate(new_velocity):
        if vel < -5:
            updated_velocity.append(-5.0)
        elif vel > 5:
            updated_velocity.append(5.0)
        else:
            updated_velocity.append(vel)

    # particle updating
    new_particle = list(particle + np.array(updated_velocity))

    # 4.outlier handling - maintain the particle and velocity within their valid ranges
    updated_particle = []
    for i in range(cur_len):
        if new_particle[i] < -10:
            updated_particle.append(-10.0)
        elif new_particle[i] > 10:
            updated_particle.append(10.0)
        else:
            updated_particle.append(round(new_particle[i], 4))
    return updated_particle, updated_velocity


def save_population(type, population, KTau_list, args, gen_no, eval_ktau=None, search_time=None, rho_list=None):
    file_name = args.save_dir+'/' + type + '_%02d.txt' % (gen_no)
    _str = save_pop(population, KTau_list, gen_no, eval_ktau, search_time, rho_list)
    with open(file_name, 'w') as f:
        f.write(_str)

def save_pop(population, KTau_list, gen_no, eval_ktau, time, rho_list):
    pop_str = []
    for id, particle in enumerate(population):
        _str = []
        _str.append('indi:%02d' % (id))
        _str.append('particle:%s' % (','.join(list(map(str, particle)))))
        if not gen_no == -1:
            _str.append('KTau:%.4f' % (KTau_list[id]))
        else:
            _str.append('KTau_list:%s' % (','.join(list(map(str, KTau_list)))))
            _str.append('rho_list:%s' % (','.join(list(map(str, rho_list)))))

        if eval_ktau:
            _str.append('eval_ktau on the target task:%s' % (eval_ktau))
        if time:
            _str.append('search time:%s' % (time))

        particle_str = '\n'.join(_str)
        pop_str.append(particle_str)
        pop_str.append('-' * 100)
    return '\n'.join(pop_str)
