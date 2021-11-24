import taichi as ti
import numpy as np
from mandelbrot import MAX_ITER



ti.init(arch=ti.gpu,default_fp=ti.f64)

n = 1000
EPS=0.000001
MAX_ITER=40

SHAPE_SCENE = (n,n)

pixels = ti.Vector.field(3, dtype=ti.f64, shape=SHAPE_SCENE)
gui = ti.GUI("Newton's Fractal", res=SHAPE_SCENE)


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])
@ti.func
def complex_cube(z):
    return ti.Vector([z[0]**3 - 3*z[0]*z[1]**2, 3*z[0]**2*z[1] - z[1]**3])
@ti.func
def complex_divide(z, w):
    # return z / w, z and w complex vectors
    return ti.Vector([(z[0]*w[0] + z[1]*w[1])/(w[0]**2 + w[1]**2), (z[1]*w[0] - z[0]*w[1])/(w[0]**2 + w[1]**2)])
# polynomials
@ti.func
def pz(z):
    return complex_cube(z) - ti.Vector([1.0,0.0])
@ti.func 
def dpz(z):
    return 3.0*complex_sqr(z)
@ti.func
def newton_method(z):
    return complex_divide(pz(z),dpz(z))
@ti.func
def complex_abs(z):
    return ti.sqrt(z[0]**2 + z[1]**2)


# 3 complex roots of x^3 - 1 = 0
roots = ti.Matrix([[1.0, 0.0], [-0.5, 0.86603],[-0.5, -0.86603]])


color_root_from_index = {0: ti.Vector([0,0,255]),
                         1: ti.Vector([0,255,0]),
                         2: ti.Vector([255,0,0]),
                         3: ti.Vector([0,0,0])}

color_root_1 = ti.Vector([0,0,255])
color_root_2 = ti.Vector([0,255,0])
color_root_3 = ti.Vector([255,0,0])
color_root_4 = ti.Vector([255,255,255])



@ti.kernel
def test(t: float):
    for _ in range(1):
        #test complex cube (result is (-81,-52))
        print(complex_cube(ti.Vector([3,-3.45]))-ti.Vector([1.0,0.0]))
        # text complex sqr (result is (-8.70,-62.1))
        print(3*complex_sqr(ti.Vector([3,-3.45])))
        # test complex divide (result is (-1.11,0.0111))
        print(complex_divide(ti.Vector([3.3,-3.4]), ti.Vector([-3.0,3.0])))

        roots = ti.Matrix([[1.0, 0.0], [-0.5, 0.86603],[-0.5, -0.86603]])
        z = ti.Vector([2.0,3.3])
        z = ti.cast(z, ti.f64)
        for o in ti.static(range(10)):
            z = z - newton_method(z)
        print(z)
        for ii in ti.static(range(3)):
            root = ti.Vector([roots[ii,0],roots[ii,1]])
            print((z - root).norm(), EPS)
            if complex_abs(z-root) < EPS:
                print(ii,z,color_root_from_index[ii],'EAE')
        
@ti.kernel
def test_paint(t: float):
        for i,j in pixels:  # Parallelized over all pixels
            coords = [((i*3.0) / n) - 2.0, ((j*3.0) / n) -1.5]
            if(i==0 and j==0):
                print(i,j,coords)
            if(i==n-1 and j==n-1):
                print(i,j,coords)
                
@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.66* ti.sin(t), ti.cos(t) * 0.02])
        z = ti.Vector([((i*3.0) / n) -2.0, ((j*3.0) / n) -1.5]) 
        iterations = 0
        not_converged = True
        while not_converged: 
            term = newton_method(z) + c 
            z-=(term) 
            not_converged = complex_abs(term) > EPS
            iterations += 1
            if(iterations > MAX_ITER):
                break
        if not not_converged:
            
            min = complex_abs(z-ti.Vector([roots[0,0],roots[0,1]]))
            index = 0
            if complex_abs(z-ti.Vector([roots[1,0],roots[1,1]])) < min:
                min = complex_abs(z-ti.Vector([roots[1,0],roots[1,1]]))
                index = 1
            if complex_abs(z-ti.Vector([roots[2,0],roots[2,1]])) < min:
                min = complex_abs(z-ti.Vector([roots[2,0],roots[2,1]]))
                index = 2

            #print(z,iterations, min)
            #  WHAT THE FUCK IS THIS

            if index==0:
                pixels[i, j] =  color_root_1 * ((MAX_ITER-iterations*0.10)/MAX_ITER) 
            elif index==1:
                pixels[i, j] = color_root_2 * ((MAX_ITER-iterations*0.10)/MAX_ITER)
            elif index==2:
                pixels[i, j] = color_root_3 * ((MAX_ITER-iterations*0.10)/MAX_ITER)

        else:
            pixels[i,j] = color_root_4 * ((MAX_ITER-iterations*0.10)/MAX_ITER)




make_video = False

if(make_video):
    result_dir = "./results"
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)



for i in range(1000):
    paint(i * 0.03)
    if not make_video:
        gui.set_image(pixels)
        gui.show()
    else:
        pixels_img = pixels.to_numpy()
        video_manager.write_frame(pixels_img)
        print(f'\rFrame {i+1}/50 is recorded', end='')

if make_video:
    print()
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')