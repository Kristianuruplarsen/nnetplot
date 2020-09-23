""" Tools for plotting neural networks in matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np

from .activations import dispacth_activation, relu, sigmoid, linear, make_node_data


class Node:
    """ A single node in a neural network.

    :param x: x-coord of node center.
    :type x: float

    :param y: y-coord of node center.
    :type y: float

    :param radius: radius of node.
    :type radius: float

    :param activation: string identifying activation, e.g. 'sigmoid' (or 
                an already instantiated activation function).
    :type activation: str

    :param special: Indicates that the layer does not have an activation function.
    :type special: bool
    """

    def __init__(self, x, y, radius, activation, special=False):
        self.x = x
        self.y = y
        self.radius = radius
        self.activation = dispacth_activation(activation)
        self.special = special

    def draw(self, axis):
        """ Draw the node on a supplied matplotlib axis.

        :param axis: matplotlib axis.
        """
        circle = plt.Circle(
            (self.x, self.y), radius=self.radius, facecolor="white", edgecolor="black"
        )
        axis.add_patch(circle)

        if not self.special:
            act = axis.plot(
                *make_node_data(self.x, self.y, self.radius, self.activation),
                color="blue",
                linewidth=5
            )
            act[0].set_clip_path(circle)

        # TODO: do special stuff with input/output nodes?
        elif self.special == "input":
            pass
        elif self.special == "output":
            pass


class Layer:
    def __init__(
        self,
        rows,
        columns,
        activation=None,
        special=False,
        xanchor=0,
        yanchor=0,
        radius=0.2,
        vspace=0.1,
        hspace=0.1,
    ):
        """ A neural network layer, generally with horizontally aligned nodes.

        :param rows: number of rows of nodes (translates to rectangle height in .draw_rect)
        :type rows: int

        :param columns: number of columns of nodes (translates to rectangle width in .draw_rect)
        :type columns: int

        :param activation: activation function in the layer.
        :type activation: str

        :param special: Special layers do not have an activation function.
        :type special: bool

        :param xanchor: x-coord of upper left corner of layer.
        :type xanchor: float

        :param yanchor: y-coord of upper left corner of layer.
        :type yanchor: float

        :param radius: radius of layer nodes.
        :type radius: float

        :param vspace: vertical spacing between nodes.
        :type vspace: float

        :param hspace: horizontal spacing between nodes. 
        :type hspace: float

        Usage::

            >>> # First set up the layers you want to plot
            >>> state = Layer(rows=1, columns=1, special = 'input')
            >>> h1 = Layer(rows=12, columns=1, activation = 'sigmoid')
            >>> action = Layer(rows = 1, columns = 1, special = 'input')
            >>> h2 = Layer(rows=12, columns=1, activation = 'sigmoid')
            >>> out = Layer(rows = 1, columns = 1, activation='sigmoid')
            >>> 
            >>> # Vertically align layers
            >>> vertical_align(state, h1, ratio = 0.5)
            >>> vertical_align(h1, action, ratio = -0.12)
            >>> vertical_align(h1, h2, ratio = 0.5)
            >>> vertical_align(h2, out, ratio = 0.5)
            >>> 
            >>> # Horizontally align layers
            >>> horizontal_align(state, h1, spacing = 1)
            >>> horizontal_align(h1, action, spacing = 0.2)
            >>> horizontal_align(h1, h2, spacing = 1)
            >>> horizontal_align(h2, out, spacing = 1)
            >>> 
            >>> # Figure setup
            >>> fig, ax = plt.subplots(figsize = (16,16))
            >>> ax.set_aspect('equal')
            >>> ax.axis('off')
            >>> 
            >>> Draw all of the network layers
            >>> state.draw_nodes(ax)
            >>> state.annotate_nodes(ax, ['$M_t$'], fontsize = 20)
            >>> h1.draw_nodes(ax)
            >>> action.draw_nodes(ax)
            >>> action.annotate_nodes(ax, ['$a_t$'], fontsize = 20)
            >>> h2.draw_nodes(ax)
            >>> out.draw_nodes(ax)
            >>> 
            >>> Draw layer connections
            >>> connect_nodes_to_nodes(layer0=state, layer1=h1, axis=ax)
            >>> connect_nodes_to_nodes(layer0=h1, layer1=h2, axis=ax)
            >>> connect_nodes_to_nodes(layer0=action, layer1=h2, axis=ax)
            >>> connect_nodes_to_nodes(layer0=h2, layer1=out, axis=ax)
        """
        self.rows = rows
        self.columns = columns
        self.activation_str = activation
        self.activation = dispacth_activation(activation)
        self.special = special
        self.xanchor = xanchor
        self.yanchor = yanchor
        self.radius = radius
        self.vspace = vspace
        self.hspace = hspace

    def inbound_node_connections(self):
        """ Connection points for inbound connections
            to layer nodes.
        """
        for x, y in self.node_centers():
            yield x - 0.2 * self.radius, y

    def outbound_node_connections(self):
        """ Connection points for outbound connections
            from layer nodes.
        """
        for x, y in self.node_centers():
            yield x + 0.2 * self.radius, y

    def inbound_rect_connections(self, pad=0.0, width=0.1):
        """ Connection points for inbound connections 
            to layer rectangle.

            :param pad: all coordinates are relative to a 
                        rectangle with this much padding.
            :type pad: float

            :param width: amount of vertical distance between connection points.
            :type width: float
        """
        coords = self.rect_coords(pad)
        x = coords["xy"][0]
        y1 = coords["xy"][1] + (0.5 - width / 2) * coords["height"]
        y2 = coords["xy"][1] + (0.5 + width / 2) * coords["height"]
        return [(x, y1), (x, y2)]

    def outbound_rect_connections(self, pad=0.0, width=0.1):
        """ Connection points for outbound connections 
            from layer rectangle.

            :param pad: all coordinates are relative to a 
                        rectangle with this much padding.
            :type pad: float

            :param width: amount of vertical distance between connection points.
            :type width: float            
        """
        coords = self.rect_coords(pad)
        x = coords["xy"][0] + coords["width"]
        y1 = coords["xy"][1] + (0.5 - width / 2) * coords["height"]
        y2 = coords["xy"][1] + (0.5 + width / 2) * coords["height"]
        return [(x, y1), (x, y2)]

    def node_centers(self):
        """ Centerpoints of every node in the layer as pairs (xi, yi)
        """
        x = self.xanchor + self.radius
        y = self.yanchor - self.radius
        for col_idx in range(self.columns):
            for row_idx in range(self.rows):
                yield (x, y)
                y -= 2 * self.radius + self.vspace
            y = self.yanchor - self.radius
            x += 2 * self.radius + self.hspace

    def rect_coords(self, pad=0):
        """ Coordinates identifying the layer rectangle.

            :param pad: Padding on all sides of the layer rectangle.
            :type pad: float
        """
        coords = {
            "xy": (self.xanchor - pad, self.yanchor + pad),
            "width": 2 * self.radius * self.columns + 2 * pad + (self.columns - 1) * self.hspace,
            "height": -(2 * self.radius * self.rows + 2 * pad + (self.rows - 1) * self.vspace),
        }
        return coords

    def draw_nodes(self, axis):
        """ Draw the layer nodes.
        
        :param axis: matplotlib axis.
        """
        for x, y in self.node_centers():
            node = Node(
                x=x, y=y, radius=self.radius, activation=self.activation, special=self.special
            )
            node.draw(axis)

    def draw_rect(self, axis, pad=0, draw_activation=True, **rectkw):
        """ Draw the layer rectangle.

            :param axis: matplotlib axis.
            
            :param pad: padding to apply on all sides of the layer rectangle.
            :type pad: float

            :param draw_activations: draw an activation in the center of the recangle.
            :type draw_activations: bool

            :param \**rectkw: Additonal options passed directly to matplotlib.pyplot.Rectangle 
        """
        otherkw = {
            k: v for k, v in rectkw.items() if k not in ["fill", "alpha", "edgecolor", "zorder"]
        }
        coords = self.rect_coords(pad=pad)
        rect = plt.Rectangle(
            **coords,
            fill=rectkw.get("fill", False),
            alpha=rectkw.get("alpha", 1.0),
            edgecolor=rectkw.get("edgecolor", "black"),
            zorder=rectkw.get("zorder", -10),
            **otherkw
        )
        axis.add_patch(rect)

        if draw_activation:
            x = self.xanchor + 0.5 * coords["width"] - pad
            y = self.yanchor + 0.5 * coords["height"] + pad

            act = axis.plot(
                *make_node_data(x, y, self.radius, self.activation), color="blue", linewidth=5
            )
            act[0].set_clip_path(rect)

        elif self.special == "input":
            pass
        elif self.special == "output":
            pass

    def annotate_nodes(self, axis, annotations, **annotationkw):
        """ Annotate nodes. Annotations are placed at the center of the node.

        :param annotations: A list of annotations. Must have the same length as the number of nodes
                           in the layer.
        :type annotations: list

        :param \**annotationkw: Additional arguments passed directly to axis.annotate
        """
        for txt, (x, y) in zip(annotations, self.node_centers()):
            axis.annotate(
                txt,
                xy=(x, y),
                verticalalignment="center",
                horizontalalignment="center",
                **annotationkw
            )

    def annotate_rect(self, axis, annotation, ypad=0, xpad=0, **annotationkw):
        """ Annotate a rect below and left aligned.

        :param annotations: Annotation string.
        :type annotations: str

        :param ypad: y-padding to the annotation position.
        :type ypad: float

        :param xpad: x-padding to the annotation position.
        :type xpad: float

        :param \**annotationkw: Additional arguments passed directly to axis.annotate        
        """
        coords = self.rect_coords()
        x, y = coords["xy"]
        axis.annotate(
            annotation,
            xy=(x - xpad, y + coords["height"] - ypad),
            verticalalignment="top",
            horizontalalignment="left",
            **annotationkw
        )


def vertical_align(layer0, layer1, ratio=0.5):
    """ Align two adjacent layers vertically.

    :param layer0: left layer
    :type layer0: nnetplot.Layer

    :param layer1: right layer
    :type layer1: nnetplot.Layer    

    :param ratio: A ratio of 0.5 aligns the center of layer0 to
                  the center of layer1. 
    :type ratio: float
    """
    y0 = layer0.yanchor
    h0 = layer0.rect_coords()["height"]
    y1 = layer1.yanchor
    h1 = layer1.rect_coords()["height"]

    layer1.yanchor = y0 - ratio * (h1 - h0)


def horizontal_align(layer0, layer1, spacing):
    """ Align two adjacent layers vertically.

    :param layer0: left layer
    :type layer0: nnetplot.Layer

    :param layer1: right layer
    :type layer1: nnetplot.Layer    

    :param spacing: spacing between the two layers.
    :type spacing: float    
    """
    x0 = layer0.xanchor
    layer1.xanchor = layer0.xanchor + spacing


def connect_nodes_to_rect(layer0, layer1, axis, inbound_kw={}):
    """ Connect a layer drawn as nodes to a layer drawn as a rectangle.

        :param layer0: left layer
        :type layer0: nnetplot.Layer

        :param layer1: right layer
        :type layer1: nnetplot.Layer    

        :param axis: matplotlib axis

        :param inbound_kw: padding and width when getting the rectangle connection points.
        :type inbound_kw: dict
    """
    for xout, yout in layer0.outbound_node_connections():
        for xin, yin in layer1.inbound_rect_connections(**inbound_kw):
            axis.plot([xout, xin], [yout, yin], color="black", zorder=-10)


def connect_nodes_to_nodes(layer0, layer1, axis):
    """ Connect a layer drawn as nodes to a layer drawn as a rectangle.

        :param layer0: left layer
        :type layer0: nnetplot.Layer

        :param layer1: right layer
        :type layer1: nnetplot.Layer    

        :param axis: matplotlib axis
    """
    for xout, yout in layer0.outbound_node_connections():
        for xin, yin in layer1.inbound_node_connections():
            axis.plot([xout, xin], [yout, yin], color="black", zorder=-10)


def connect_rect_to_nodes(layer0, layer1, axis, outbound_kw={}):
    """ Connect a layer drawn as nodes to a layer drawn as a rectangle.

        :param layer0: left layer
        :type layer0: nnetplot.Layer

        :param layer1: right layer
        :type layer1: nnetplot.Layer    

        :param axis: matplotlib axis

        :param outbound_kw: padding and width when getting the rectangle connection points.
        :type outbound_kw: dict
    """
    for xout, yout in layer0.outbound_rect_connections(**outbound_kw):
        for xin, yin in layer1.inbound_node_connections():
            axis.plot([xout, xin], [yout, yin], color="black", zorder=-10)


def connect_rect_to_rect(layer0, layer1, axis, outbound_kw={}, inbound_kw={}):
    """ Connect a layer drawn as nodes to a layer drawn as a rectangle.

        :param layer0: left layer
        :type layer0: nnetplot.Layer

        :param layer1: right layer
        :type layer1: nnetplot.Layer    

        :param axis: matplotlib axis

        :param outbound_kw: padding and width when getting the outbound rectangle connection points.
        :type outbound_kw: dict

        :param inbound_kw: padding and width when getting the inbound rectangle connection points.
        :type inbound_kw: dict
    """
    for xout, yout in layer0.outbound_rect_connections(**outbound_kw):
        for xin, yin in layer1.inbound_rect_connections(**inbound_kw):
            axis.plot([xout, xin], [yout, yin], color="black", zorder=-10)

