#!-*-coding:utf-8-*-
import imageio
import numpy as np
import seaborn as sns
import modules as mdl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sns.set()

# 真の関数
def y( x ):
    return 0.06*x**3 - 0.4 * x ** 2 + 2 * x + np.sin( np.pi * x )

def process( x0 , x1 , y0 , kernel ):
    # Gram Matrix
    k00 = kernel( *np.meshgrid( x0 , x0 ) )
    k01 = kernel( *np.meshgrid( x0 , x1 , indexing='ij' ) )
    k10 = k01.T
    k11 = kernel( *np.meshgrid( x1 , x1 ) )

    k00_inv = np.linalg.inv( k00 )

    u = k10.dot( k00_inv.dot( y0 ) )
    sigma = k11 - k10.dot( k00_inv.dot( k01 ) )

    return u , sigma

n = 100
x0 = np.random.uniform( 0 , 10 , n )
y0 = y( x=x0 ) + np.random.normal( 0 , 2 , n )
x1 = np.linspace( -1 , 11 , 101 )
kernel = mdl.RBF2D( 8 , 0.5 , 3.5 )

ims = []
fig , ax = plt.subplots()
def update( i ):
    plt.cla()
    u , sigma = process( x0=x0[:i] , x1=x1 , y0=y0[:i] , kernel=kernel )
    std = np.sqrt( sigma.diagonal() )
    ax.scatter( x0[:i] , y0[:i] )
    ax.plot( x1 , u )
    ax.plot( x1 , y( x=x1 ) , '--' )
    ax.fill_between( x1 , u - std , u + std , alpha=0.2 , color="b" )

anim = FuncAnimation ( fig , update , frames=n )
#anim.save("gausian_fit_animation.gif")
plt.show()
