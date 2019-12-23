from sklearn.svm import SVR
import matplotlib.pyplot as plt


plt.figure(figsize=(7.48,7.48))

ax1 = plt.subplot2grid((3,5), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,5), (0,3), colspan=2,aspect='equal')
ax3 = plt.subplot2grid((3,5), (1,0), colspan=3)
ax4 = plt.subplot2grid((3,5), (1,3), colspan=2,aspect='equal')
ax5 = plt.subplot2grid((3,5), (2,0), colspan=3)
ax6 = plt.subplot2grid((3,5), (2,3), colspan=2,aspect='equal')

plt.tight_layout()
plt.show()