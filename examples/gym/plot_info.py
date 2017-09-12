import re
import sys
import numpy as np
import matplotlib.pyplot as pyplot

# example line
# INFO:chainerrl.experiments.train_agent:outdir:dqn_out/20170912T082308.938788 step:15 episode:0 R:0.015000000000000006
pattern_step = re.compile(r"step:(\d+)")
pattern_reward = re.compile(r"R:([01]\.\d+)")

result = np.array([[0, 0]])
pyplot.ion()
pyplot.show(block=False)


# https://stackoverflow.com/a/24272092
class PyPlotDynamic():
    def __init__(self):
        # Set up plot
        self.figure, self.ax = pyplot.subplots()
        self.lines, = self.ax.plot([], [], '+-')
        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        # Other stuff
        self.ax.grid()

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Example
    def __call__(self, xdata, ydata):
        self.on_running(xdata, ydata)


d = PyPlotDynamic()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    matched_step = re.search(pattern_step, line)
    matched_reward = re.search(pattern_reward, line)
    if not matched_reward or not matched_step:
        continue
    step = int(matched_step.group(1))
    reward = float(matched_reward.group(1))
    print("{}:{}".format(step, reward))
    result = np.append(result, [[step, reward]], axis=0)
    d(result[:, 0], result[:, 1])
