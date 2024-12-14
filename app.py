import tkinter as tk
import numpy as np

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
    
    draw_thickness = 150

    canvas_list = []

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



    def parse_click(event):
        draw(event.x,event.y,255)

    def draw(event_x,event_y, weight):
        x = event_x
        y = event_y 

        mouse_x = (x - canvas_x_offset) // pixel_size 
        mouse_y = (y - canvas_y_offset) // pixel_size 


        if 0 <= mouse_x < n_pixels and 0 <= mouse_y < n_pixels: #In bounds
            
            target = canvas_list[mouse_y + mouse_x* n_pixels]
            old_fill = 255 - int(canvas.itemcget(target,"fill")[1:3], base=16) # 0 is white, 255 is black
            if  weight > old_fill:
                new_fill = old_fill + weight #Add on black colouring
                if new_fill > 255: #255 cap
                    new_fill = 255
                string_code = "#{}".format(hex(255 - new_fill)[2:] * 3)
                canvas.itemconfig(target, fill=string_code)  # Change the fill color to black

                if weight - draw_thickness > 0: #Shade nearby pixels
                    draw(event_x-pixel_size,event_y, weight - draw_thickness) #Left
                    draw(event_x+pixel_size,event_y,weight - draw_thickness) #Right

                    draw(event_x,event_y -pixel_size ,weight - draw_thickness) #Down
                    draw(event_x,event_y +pixel_size ,weight - draw_thickness) #Up
                    


            # Neural network array value
            # drawing_canvas[mouse_y, mouse_x] = 255

        
    canvas.bind("<B1-Motion>", parse_click)
    

    initalise_canvas()
    window.mainloop()


if __name__ == "__main__":
    main()
