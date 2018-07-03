from scipy.optimize import fsolve
import pylab
import numpy


def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0)


result = findIntersection(numpy.sin, numpy.cos, 0.0)
x = numpy.linspace(0.001, 2, 50)
pylab.plot(x, [1 / numpy.sin(y) for y in x])
pylab.show()
