import tkinter as tk
import numpy as np
from main import Neural_Network

def main():
    window = tk.Tk()
    window.title("Draw")

    #Canvas is square
    n_pixels = 30 


    canvas_size = 15 #Affects x and y
    canvas_size_y = n_pixels * canvas_size 
    canvas_size_x = canvas_size_y * 2 

    canvas_x_offset = 30
    canvas_y_offset = 20
    window.focus_force()

    pixel_size = canvas_size_y // n_pixels - 1 #-1 to fit on window

    canvas = tk.Canvas(window, width=canvas_size_x, height=canvas_size_y, bg="white")
    canvas.pack()
    
    drawing_canvas = np.zeros((n_pixels,n_pixels), dtype=np.uint8)
    
    draw_intensity = 90
    draw_thickness = 3

    canvas_list = []

    class grid:
        def __init__(self,rect,):
            self.rect = rect
            self.root = "goop"




    def initalise_canvas():
        for i in range(n_pixels):
            for j in range(n_pixels):
                #Draw Rectangle using pixel size and adding any offset
                rect = canvas.create_rectangle(
                    i * pixel_size + canvas_x_offset, j * pixel_size + canvas_y_offset,
                    (i + 1) * pixel_size + canvas_x_offset, (j + 1) * pixel_size + canvas_y_offset,
                    fill="#FFFFFF" , outline="purple"
                )
                #Store coordinates to list
                canvas_list.append(rect)





    def calculateWeight(hex): # 0 is white, 255 is black, Arguement is hex string
            old_fill = 255 - int(hex[1:3], base=16) 
            return old_fill
 
    def addWeights(weight1, weight2):
        weight = weight1 + weight2
        weight = min(weight , 255)
        weight = max(weight, 0)
        return weight

    def calculateColour(weight): #Returns String hex code eg #FFFFFF
        #Todo, add color options while keeping weight functionalilty the same
        hex_str =  hex(255-weight)[2:4]
        if len(hex_str) == 1: #Example: #999 -> #090909
            hex_str = "0"+hex_str
        return "#"+hex_str * 3 






    def parse_R_click(event):
        draw(event.x,event.y,-255)

    def parse_L_click(event):
        draw(event.x,event.y,draw_intensity)


    def erase(event):
        for square in canvas_list:
            canvas.itemconfig(square, fill="#FFFFFF")  # Clear all to white


    






    def call_network(inputs):
        nn = Neural_Network()
        nn.initialise_network(inputs)
        nn.run_network()
        nn.propagate_backward()




    def export(event):
        pixel_arr = [] #1D list of inputs
        for pixel in canvas_list:
            pixel_arr.append(round((calculateWeight(canvas.itemcget(pixel, "fill")) / 255.0), 2)) #Append a normalised value pixel value, 0 is white, 1 is black

        call_network(pixel_arr)


    def draw(event_x,event_y, weight):
        x = event_x
        y = event_y 

        mouse_x = (x - canvas_x_offset) // pixel_size 
        mouse_y = (y - canvas_y_offset) // pixel_size 


        if 0 <= mouse_x < n_pixels and 0 <= mouse_y < n_pixels: #In bounds
            
            target = canvas_list[mouse_y + mouse_x* n_pixels]


            new_weight = addWeights(calculateWeight(canvas.itemcget(target, "fill")), weight)
            new_color = calculateColour(new_weight)
            canvas.itemconfig(target, fill=new_color)  # Change the fill color to black

            if draw_thickness > 0:
                weight_loss = (draw_intensity // draw_thickness)
                spread_weight = weight - weight_loss

                if spread_weight > 0 or (weight < 0 and -(255+weight) // weight_loss < draw_thickness): #Shade nearby pixels
                    draw(event_x-pixel_size,event_y, spread_weight) #Left
                    draw(event_x+pixel_size,event_y,spread_weight) #Right

                    draw(event_x,event_y -pixel_size ,spread_weight) #Down
                    draw(event_x,event_y +pixel_size ,spread_weight) #Up






    #Left click to Draw, command + left click = erase   
    canvas.bind("<B1-Motion>", parse_L_click)
    canvas.bind("<Button-1>", parse_L_click)

    #Right click to erase
    canvas.bind("<B2-Motion>", parse_R_click)
    canvas.bind("<Button-2>", parse_R_click)

    #R to erase
    canvas.bind("<r>", erase)

    #O to export
    canvas.bind("<o>", export)
    

    canvas.focus_set()

    initalise_canvas()
    window.mainloop()


if __name__ == "__main__":
    main()
