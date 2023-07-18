import cv2
import pdf2image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import shapely


def read_pdf(path_to_pdf):
    operation_map = pdf2image.convert_from_path(path_to_pdf)
    operation_map = operation_map[0]

    return operation_map


def crop_image(img):
    x_start, y_start, x_end, y_end = 50, 30, 2100, 1288  # Define the coordinates of the cropping region
    cropped_image = img.crop((x_start, y_start, x_end, y_end))
    gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)
    return gray_image


def extract_gradients(img):

    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    return edges


def get_max_index(array):
    start_x = array[:, :200].sum(axis=0).argmax()
    end_x = array[:, -200:].sum(axis=0).argmax() + array.shape[1] - 200
    start_y = array[:200, :].sum(axis=1).argmax()
    end_y = array[-200:, :].sum(axis=1).argmax() + array.shape[0] - 200
    return start_x, start_y, end_x, end_y


def map_axes(start_x, start_y, end_x, end_y, x_data, y_data):

    mapped_x_data = start_x + (end_x - start_x) * (x_data-0.1)/0.9
    mapped_y_data = end_y + (start_y - end_y)*(y_data-1.0)/4.0

    return mapped_x_data, mapped_y_data


def plot_separation_boundary(start_x, start_y, end_x, end_y):
    p1 = list(map_axes(start_x, start_y, end_x, end_y, 0.2, 1.8))
    p2 = list(map_axes(start_x, start_y, end_x, end_y, 0.24, 2.06))
    p3 = list(map_axes(start_x, start_y, end_x, end_y, 0.25, 2.31))
    p4 = list(map_axes(start_x, start_y, end_x, end_y, 0.29, 2.66))
    p5 = list(map_axes(start_x, start_y, end_x, end_y, 0.424, 3.05))
    p6 = list(map_axes(start_x, start_y, end_x, end_y, 0.52, 3.56))
    p7 = list(map_axes(start_x, start_y, end_x, end_y, 0.6, 4.1))
    p8 = list(map_axes(start_x, start_y, end_x, end_y, 0.675, 4.68))
    p9 = list(map_axes(start_x, start_y, end_x, end_y, 0.7, 4.85))
    critical_boundary = shapely.LineString([p1, p2, p3, p4, p5, p6, p7, p8, p9])
    return critical_boundary


def animate_data(start_x, start_y, end_x, end_y, data_plot, scatter, critical_boundary):
    def animate(i):
        nonlocal data_plot

        new_data = np.random.rand(1, 2)
        if new_data[0, 0] < 0.1:
            new_data[0, 0] = new_data[0, 0] + 0.1
        new_data[0, 1] = new_data[0, 1] + 3
        data_plot = np.vstack((data_plot, new_data))

        indices = np.arange(max(0, i - 2), i + 1)
        x_data = []
        y_data = []
        for index in indices:
            x, y = map_axes(start_x, start_y, end_x, end_y, data_plot[index, 0], data_plot[index, 1])
            x_data.append(x)
            y_data.append(y)
        sizes = [10, 25, 40]
        scatter.set_offsets(np.c_[x_data, y_data])
        scatter.set_sizes(sizes)
        last_dot = shapely.Point(x_data[-1], y_data[-1])
        dist = critical_boundary.distance(last_dot)
        orientation = 0.5*(data_plot[index, 1] - 1.8) - 3.05*(data_plot[index, 0] - 0.2)
        print(data_plot[index, 1], data_plot[index, 0])
        print(orientation)
        if orientation > 0:
            scatter.set_color('red')
        elif dist < 150:
            scatter.set_color('yellow')
        else:
            scatter.set_color('green')

        return scatter,
    return animate


def main():
    pdf_path = 'turbolader_verdichterkennfeld.pdf'
    op_map = read_pdf(pdf_path)
    img_crop = crop_image(op_map)

    img_edges = extract_gradients(img_crop)
    start_x, start_y, end_x, end_y = get_max_index(img_edges)
    critical_boundary = plot_separation_boundary(start_x, start_y, end_x, end_y)
    x_c, y_c = critical_boundary.xy

    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.imshow(img_crop, cmap='gray')
    ax.plot(x_c, y_c)

    data = np.array([[0.6, 2.0], [0.5, 3.0]])
    animation = FuncAnimation(fig, animate_data(start_x, start_y, end_x, end_y, data, scatter, critical_boundary),
                              frames=15, interval=1000, blit=True, repeat=True)
    plt.show()

    ''' x, y = map_axes(start_x, start_y, end_x, end_y, 0.6, 2.0)

    fig, ax = plt.subplots()
    plt.ion()
    ax.imshow(img_crop, cmap='gray')
    ax.plot(x, y, 'ro')
    ax.plot(x_c, y_c)
    plt.show()
    plt.pause(2)

    x, y = map_axes(start_x, start_y, end_x, end_y, 0.5, 3.0)
    print(x, y)
    fig, ax = plt.subplots()
    
    plt.ion()
    ax.imshow(img_crop, cmap='gray')
    ax.plot(x, y, 'ro')
    plt.show()
    plt.pause(5)'''


if __name__ == '__main__':
    main()
