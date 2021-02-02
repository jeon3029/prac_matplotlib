import matplotlib.pyplot as plt, numpy as np
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import proj3d
from matplotlib import animation
from IPython.display import HTML

def visualize3DData(X, scale, cmap):
    """Visualize data in 3d plot with popover next to mouse position.
    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = []
    for i in X[:,2]:
        if float(i) >= 10:
            colors.append('g')
        elif float(i) < 10 and float(i) > 0:
            colors.append('grey')
        else:
            colors.append('r')
    # im = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3], s=X[:, 4] * scale, cmap=cmap, alpha=1, picker=True)
    im = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=X[:, 3]*scale, cmap = cmap, alpha = 0.8,
                    color = colors,
                    picker = True
                    )
    ax.set_xlabel('원가율')
    ax.set_ylabel('할인율')
    ax.set_zlabel('영업이익률')

    # cbar = fig.colorbar(im)
    # cbar.ax.set_ylabel('OBJECTIVE 4')
    #
    # objs = X[:, 1]/10
    #
    # max_size = np.amax(objs) * scale / 32.0
    # min_size = np.amin(objs) * scale / 4.5
    # handles, labels = ax.get_legend_handles_labels()
    # display = (0, 1, 2)
    #
    # size_max = plt.Line2D((0, 1), (0, 0), color='k', marker='o', markersize=max_size, linestyle='')
    # size_min = plt.Line2D((0, 1), (0, 0), color='k', marker='o', markersize=min_size, linestyle='')
    # legend1 = ax.legend([handle for i, handle in enumerate(handles) if i in display] + [size_max, size_min],
    #                     [label for i, label in enumerate(labels) if i in display] + ["%.2f" % (np.amax(objs)),
    #                                                                                  "%.2f" % (np.amin(objs))],
    #                     labelspacing=1.5, title='OBJECTIVE 5', loc=1, frameon=True, numpoints=1, markerscale=1)

    def distance(point, event):
        """Return distance between mouse position and given data point
        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)

    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance(X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)

    def annotatePlot(X, index):
        """Create popover label in 3d chart
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        annotatePlot.label = plt.annotate("index: %d" % index,
                                          xy=(x2, y2), xytext=(-20, 20), textcoords='offset points', ha='right',
                                          va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        fig.canvas.draw()

    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot(X, closestIndex)

    # replace '-'
    # plt.gca().set_zticklabels([i.get_text().replace('−', '$-$') for i in ax.get_zticklabels()])
    # doesn't work


    # set label
    labels= ['홍대','건대','강남','세종','마포']
    # labels=[]
    # for i in range(len(X)):
    #     labels.append(''.join(chr(ord('A')+i)))
    for i in range(len(X)):  # plot each point + it's index as text above
        x = X[i, 0]
        y = X[i, 1]
        z = X[i, 2]
        label = labels[i]
        ax.scatter(x, y, z, color='b')
        ax.text(x, y+2, z, '%s' % (label), size=20, zorder=1, color='k')





    # set plane
    X = np.arange(0, 35, 0.25)
    Y = np.arange(0, 35, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = X*0
    # surf = ax.plot_surface(X, Y, Z,
    #                        # cmap=red,
    #                        color='r',
    #                        linewidth=0,
    #                        # antialiased=False,
    #                        alpha=0.5
    #                        )
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.plot()
    # fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    # plt.legend()


    # plt.show()
    def animate(frame):
        ax.view_init(30, frame / 4)
        plt.pause(.001)
        return fig
    anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)
    HTML(anim.to_html5_video())


import matplotlib.font_manager as fm

font_location = 'C:\Windows\Fonts\malgun.ttf'
# font_location = r'C:\Users\1327964\Downloads\NanumFontSetup_TTF_GOTHIC\NanumGothicBold.ttf'
# font_location = r'C:\Users\1327964\AppData\Local\Microsoft\Windows\Fonts\NanumGothicCoding-Bold.ttf'


font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font', family=font_name)
# plt.rcParams["font.serif"] = "cmr10"

if __name__ == '__main__':
    import seaborn

    # X=np.loadtxt('1000M_thined.obj')*-1
    X = np.array([[20, 20, 5,100],
                  [10, 30, 10,200],
                  [15, 27, 15,300],
                  [25, 30, -5,150],
                  [17, 26, 2,1000]]
                 )

    scale = 5
    cmap = cm.get_cmap("Spectral")
    # cmap=plt.cm.spectral
    visualize3DData(X, scale, cmap)
