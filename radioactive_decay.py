import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd
import math as m
import random

from pathlib import Path

def generate_path(home_folder = str(Path.home()), subfolder = '/Documents/', basename = 'output', extension = 'txt'):
    # creates the path to store the data. Note that the data is not stored in the code repo directory.
    # uses the method Path.home() to find the home directory in any OS
    output_folder = home_folder + subfolder # appends a subdirectory within it.
    filename = basename + '.' + extension# defines the filename the output is to be saved in
    output_path = output_folder + filename # creates the output path
    return output_path

def generate_square_crystal(x=5, y=5):
    xs = np.arange(0, x, 1.0)
    ys = np.arange(0, y, 1.0)
    x, y = np.meshgrid(xs, ys)
    crystal = pd.DataFrame(np.array([x.flatten(), y.flatten()]).T, columns=["x", "y"])
    crystal["exists"] = True
    return crystal


def calculate_decay_constant(half_life):
    # tau = 1/ lambda
    # half_life = tau*ln(2)
    # lambda = ln(2)/half_life
    decay_constant = m.log(2)/half_life
    return decay_constant


def proportion_remaining(half_life, time_step):
    decay_constant = calculate_decay_constant(half_life)
    remaining = m.e**(-decay_constant*time_step)
    return remaining


def execute_random_decay(existence, half_life, time_step):
    if existence:
        remaining = proportion_remaining(half_life, time_step)
        result = (random.uniform(0, 1) < remaining)
    else:
        return False
    # print("before = {}, after = {}".format(existence,result))
    return result


def execute_time_step(crystal, half_life, time_step=1):
    crystal["exists"] = crystal["exists"].apply(execute_random_decay, args=(half_life, time_step))
    # print(crystal["exists"])



def get_count(crystal):
    try:
        count = crystal["exists"].value_counts()[True]
    except KeyError:
        count = 0
    return count


def get_existing(crystal):
    existing = crystal.loc[crystal["exists"]]
    return existing


def reset_plots(crystal, parents, daughters, trend, ideal):
    crystal["exists"] = True
    existing = get_existing(crystal)
    daughters.set_data(crystal["x"], crystal["y"])
    parents.set_data(existing["x"], existing["y"])
    trend.set_data([], [])
    ideal.set_data([], [])


def line_append(line, x, y):
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    if x_data is None:
        x_data = []
    x_data.append(x)

    if y_data is None:
        y_data = []
    y_data.append(y)

    line.set_data(x_data, y_data)


def calc_ideal(time, crystal, half_life):
    initial_count = len(crystal)
    current_count = initial_count * proportion_remaining(half_life, time)
    # print(current_count)
    return current_count


def animate(time, crystal, half_life, time_step, daughters, parents, trend, ideal):
    if time == 0:
        reset_plots(crystal, parents, daughters, trend, ideal)

    existing = get_existing(crystal)

    line_append(trend, time, len(existing))
    line_append(ideal, time, calc_ideal(time, crystal, half_life))
    # print(ideal.get_xdata(), ideal.get_ydata())

    daughters.set_data(crystal["x"], crystal["y"])
    parents.set_data(existing["x"], existing["y"])

    execute_time_step(crystal, half_life, time_step)
    return daughters, parents, trend, ideal


def main():
    x = 8
    y = 12
    crystal = generate_square_crystal(x, y)
    existing = get_existing(crystal)


    half_life = 5

    run_time = 15
    fps = 1
    frame_count = run_time * fps + 1
    interval = max([int(1000 / fps), 1])  # returns the interval in milliseconds between frames, minimum 1
    time_step = interval / 1000
    times = np.linspace(0., run_time, int(frame_count))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Simulation of radioactive decay by random processes\n'
                 'in a crystal of {} atoms with a half-life of {}s'.format((y*x), half_life))
    daughters, = ax1.plot(crystal["x"], crystal["y"], "o", markersize=20, color="silver", label="Inert Atoms")
    parents, = ax1.plot(existing["x"], existing["y"], "o", markersize=20, color="green", label="Radioactive Atoms")
    ax1.legend(handles=[daughters, parents], bbox_to_anchor=(0, 0), loc='upper left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    trend, = ax2.plot([], [], "green", linewidth=3, label="Remaining Atoms")
    ideal, = ax2.plot([], [], "red", label="Expectation Value")
    ax2.set_xlim(0, run_time)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, x*y)
    ax2.set_ylabel('Count')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.legend()

    ani = animation.FuncAnimation(fig, animate, times, interval=interval, blit=True,
                                  fargs=(crystal, half_life, time_step, daughters, parents, trend, ideal))
    filename = generate_path(basename = 'radioactive_decay', extension = 'gif')
    ani.save(filename=filename, writer="pillow", fps=fps)
    plt.show()


if __name__ == "__main__":
    main()
