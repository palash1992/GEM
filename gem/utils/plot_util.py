import matplotlib
import matplotlib.pyplot as plt
import random

def get_node_color(node_community):
    cnames = [item[0] for item in matplotlib.colors.cnames.iteritems()]
    node_colors = [cnames[c] for c in node_community]
    return node_colors

def plot(x_s, y_s, fig_n, x_lab, y_lab, file_save_path, title, legendLabels=None, show=False):
	plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
	markers = ['o', '*', 'v', 'D', '<' , 's', '+', '^', '>']
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	series = []
	plt.figure(fig_n)
	i = 0
	for i in range(len(x_s)):
		# n_points = len(x_s[i])
		# n_points = int(n_points/10) + random.randint(1,100)
		# x = x_s[i][::n_points]
		# y = y_s[i][::n_points]
		x = x_s[i]
		y = y_s[i]
		series.append(plt.plot(x, y, color=colors[i], linewidth=2, marker=markers[i], markersize=8))
		plt.xlabel(x_lab, fontsize=16, fontweight='bold')
		plt.ylabel(y_lab, fontsize=16, fontweight='bold')
		plt.title(title, fontsize=16, fontweight='bold')
	if legendLabels:
		plt.legend([s[0] for s in series], legendLabels)
	plt.savefig(file_save_path)
	if show:
		plt.show()

def plot_ts(ts_df, plot_title, eventDates, eventLabels=None, save_file_name=None, xLabel=None, yLabel=None, show=False):
	ax = ts_df.plot(title=plot_title, marker = '*', markerfacecolor='red', markersize=10, linestyle = 'solid')
	colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k']
	if not eventLabels:
		for eventDate in eventDates:
			ax.axvline(eventDate, color='r', linestyle='--', lw=2) # Show event as a red vertical line
	else:
		for idx in range(len(eventDates)):
			ax.axvline(eventDates[idx], color=colors[idx], linestyle='--', lw=2, label=eventLabels[idx]) # Show event as a red vertical line
			ax.legend()
	if xLabel:
		ax.set_xlabel(xLabel, fontweight='bold')
	if yLabel:
		ax.set_ylabel(yLabel, fontweight='bold')
	fig = ax.get_figure()
	if save_file_name:
		fig.savefig(save_file_name, bbox_inches='tight')
	if show:
		fig.show()
