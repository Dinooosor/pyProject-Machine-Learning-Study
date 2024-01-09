import numpy as np
import matplotlib.pyplot as plt
import scienceplots
def model(x, p):
    return x ** (2 * p + 2) / (2 + x ** (2 * p))
x = np.linspace(0.75, 1.25, 201)
# with plt.style.context(['science', 'no-latex']):
#     fig, ax = plt.subplots()
#     for p in [10, 15, 20, 30, 50, 100]:
#         ax.plot(x, model(x, p), label=p)
#     ax.legend(title='Order')
#     ax.set(xlabel='Voltage (mV)')
#     ax.set(ylabel='Current (μA)')
#     ax.autoscale(tight=True)
#     fig.savefig('fig1.png', dpi=300)
with plt.style.context(['science', 'ieee','no-latex']):
    fig, ax = plt.subplots()
    for p in [10, 20, 50]:
        ax.plot(x, model(x, p), label=p)
    ax.legend(title='Order')
    ax.set(xlabel='Voltage (mV)')
    ax.set(ylabel='Current (μA)')
    ax.autoscale(tight=True)
    fig.savefig('fig2.png', dpi=300)
# with plt.style.context(['science','ieee', 'grid', 'no-latex']):
#     fig, ax = plt.subplots()
#     for p in [10, 20, 50]:
#         ax.plot(x, model(x, p), label=p)
#     ax.legend(title='Order')
#     ax.set(xlabel='Voltage (mV)')
#     ax.set(ylabel='Current (μA)')
#     ax.autoscale(tight=True)
#     fig.savefig('fig3.png', dpi=300)
