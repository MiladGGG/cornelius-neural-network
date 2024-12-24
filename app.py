import tkinter as tk
from tkinter import PhotoImage 
from PIL import Image, ImageTk

import numpy as np
from main import Neural_Network

def main():
    window = tk.Tk()
    window.title("Cornelius | Neural Network")
    window.resizable(False, False)


    #Canvas is square
    n_pixels = 30 


    canvas_size = 15 #Affects x and y
    canvas_size_y = n_pixels * canvas_size 
    canvas_size_x = canvas_size_y * 1.05 

    canvas_x_offset = 30
    canvas_y_offset = 20
    window.focus_force()

    pixel_size = canvas_size_y // n_pixels - 1 #-1 to fit on window

    # frame holding BOTH left and right
    main_frame = tk.Frame(window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # drawing Canvas
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(left_frame, width=canvas_size_x, height=canvas_size_y, bg="white")
    canvas.pack()

    drawing_canvas = np.zeros((n_pixels, n_pixels), dtype=np.uint8)



    def train_click(i):
        if nn.hasRun:
            back_propagate(i)
        else:
            log("ERROR: Neural Network has not been run yet. Cannot train.")


    def save_click(i):
        nn.save_model()
        log("Model saved sucessfully")
        

    def load_click(i):
        e = nn.load_model()
        if e == 0:
            i.config(state="disabled")
            log("Loaded saved model from /trained_model directory")
        elif e == 1:
            log("ERROR: Model is missing from /trained_model directory. Cannot load.")




    # Right frame
    right_frame = tk.Frame(main_frame, width=250, height=470)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
    right_frame.pack_propagate(False)

    right_label = tk.Label(right_frame, text="Draw Predictions", font=("Andale Mono",20))
    right_label.pack(padx=10, pady=10)

    image_files = ["images/smile.png","images/star.png","images/heart.png","images/man.png","images/triangle.png"]
    prediction_texts = []

    row_frames = []
    for i in range(5):

        row_frame = tk.Frame(right_frame)
        row_frame.pack(anchor="w", pady=10)
        row_frames.append(row_frame)  


        img = Image.open(image_files[i])
        img = img.resize((60, 60))
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(row_frame, text="0.00%", image=img_tk, compound="right")
        label.image = img_tk  # garbage collection
        label.grid(row=0, column=0, padx=5, sticky="w")  



        button = tk.Button(row_frame, text="Train", command=lambda i=i: train_click(i))
        button.grid(row=0, column=1, padx=5, sticky="e")  


        prediction_texts.append(label)

    # Options frame
    options_frame = tk.Frame(main_frame, width=200, height=canvas_size_y)
    options_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
    options_frame.grid_propagate(False)
    #Drawing Controls
    title = tk.Label(options_frame, text="Controls", font=("Arial",18))
    title.grid(row=1, column=0, padx=0, pady=10)

    cmd_label = tk.Label(options_frame, text="• Left-click to draw\n• Right-click to erase\n• 'R' to reset", font=("Arial",14))
    cmd_label.grid(row=2, column=0, padx=0, pady=0)



    #Slider Intensity
    label = tk.Label(options_frame, text="Draw Intensity")
    label.grid(row=3, column=0, padx=0, pady=(50,0),sticky="n")
    intensity_slider = tk.Scale(options_frame, from_=0, to=255, orient="horizontal")
    intensity_slider.grid(row=4, column=0, padx=0,pady=0)

    #Slider thickness
    label = tk.Label(options_frame, text="Draw Thickness")
    label.grid(row=5, column=0, padx=50, pady=0)
    thickness_slider = tk.Scale(options_frame, from_=1, to=4, orient="horizontal")
    thickness_slider.grid(row=6, column=0, padx=50)

    #default values
    intensity_slider.set(200)
    thickness_slider.set(2)
    


    #Load model button
    button = tk.Button(options_frame, text="Load Model",state="normal", command=lambda : load_click(button))
    button.grid(row=10, column=0, padx=0,pady=(30,15)) 







    #Log
    log_Title = tk.Label(options_frame, text="Neural Network Log:", font=("Arial",18))
    log_Title.grid(row=100, column=0, padx=0, pady=(0,0), sticky="s")

    log_label = tk.Label(options_frame, text="App launched", font=("Andale Mono",15),wraplength=180)
    log_label.grid(row=101, column=0, padx=(10,0), pady=0, sticky="w")


    #Display to log
    def log(text):
        log_label.config(text=text)

        if text.startswith("ERROR:"):
            log_label.config(fg="red")
        elif text.startswith("WARNING:"):
            log_label.config(fg="yellow")
        else:
            log_label.config(fg=log_Title.cget("fg"))






    index_dict = {0: "Smiley",
              1: "Star",
              2: "Heart",
              3: "Person",
              4: "Triangle"}




    nn = Neural_Network()
    nn.initialise_network()

    load_click(button)

    canvas_list = []
    drawn_list = []
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

    def calculateColour(weight,colour_index, probability): #Returns String hex code eg #FFFFFF

        hex_str =  hex(255-weight)[2:4] #Calculate plain colour
        
        #certainty_str = hex(int((255- (probability/100) *255))  )[2:4] #Calculate hue
        certainty_str = "00"
        if probability > 0:
            certainty_str = "00" 


        if len(hex_str) == 1: #Solves bug with hex formatting
            hex_str = "0"+hex_str
        if len(certainty_str) == 1: #Example: #999 -> #090909
            certainty_str = "0"+certainty_str

        if colour_index == "black": #Default black
            return "#"+hex_str * 3 

        if weight > 0:
            if colour_index == 0: #Green
                return "#"+hex_str * 3

            if colour_index == 2: #Red
                return "#"+hex_str + certainty_str *2






    def parse_R_click(event):
        draw(event.x,event.y,-255)

    def parse_L_click(event):
        draw(event.x,event.y,intensity_slider.get())


    def erase(event):
        for square in canvas_list:
            canvas.itemconfig(square, fill="#FFFFFF")  # Clear all to white
        export()
        log("Erased canvas sucessfully")


    






    def change_colours(index,val):
        
        if(val > 40):
            for r in canvas_list:
                new_colour = calculateColour(calculateWeight(canvas.itemcget(r,"fill")) ,index,val)
                canvas.itemconfig(r, fill=new_colour)




    def call_network(inputs):
        nn.run_network(inputs)
        log("Network ran sucessfully")
        for i in range(len(prediction_texts)): #Update UI text
            text_string = "%.2f%c"%(nn.probabilities[0][i] *100,'%')
            prediction_texts[i].config(prediction_texts[i], text=text_string)

        max_val = max(nn.probabilities[0])
        index = np.where(nn.probabilities[0] == max(nn.probabilities[0]))[0][0]
        change_colours(index, max_val * 100)




    def back_propagate(i):
        #Get true value from train button click
        true_array = np.array([0,0,0,0,0]) #One hot encoded
        true_array[i] = 1 #Index the correct value
        nn.true_values = true_array #Set within memory

        nn.propagate_backward()
        nn.save_model()

        
        log("Trained and saved model: %s"%(index_dict[i]))




    def export():
        pixel_arr = [] #1D list of inputs
        for pixel in canvas_list:
            pixel_arr.append(round((calculateWeight(canvas.itemcget(pixel, "fill")) / 255.0), 2)) #Append a normalised value pixel value, 0 is white, 1 is black

        call_network(pixel_arr)


    def draw(event_x,event_y, weight):
        if weight == 0:
            return 0
        x = event_x
        y = event_y 
        mouse_x = (x - canvas_x_offset) // pixel_size 
        mouse_y = (y - canvas_y_offset) // pixel_size 
        draw_intensity = intensity_slider.get()
        draw_thickness = thickness_slider.get()

        if 0 <= mouse_x < n_pixels and 0 <= mouse_y < n_pixels: #In bounds
            
            target = canvas_list[mouse_y + mouse_x* n_pixels]


            new_weight = addWeights(calculateWeight(canvas.itemcget(target, "fill")), weight)
            new_color = calculateColour(new_weight,"black",0)
            canvas.itemconfig(target, fill=new_color)  # Change the fill color to black

            if draw_thickness > 0:
                weight_loss = (draw_intensity // draw_thickness)
                spread_weight = weight - weight_loss

                if spread_weight > 0 or (weight < 0 and -(255+weight) // weight_loss < draw_thickness): #Shade nearby pixels
                    draw(event_x-pixel_size,event_y, spread_weight) #Left
                    draw(event_x+pixel_size,event_y,spread_weight) #Right

                    draw(event_x,event_y -pixel_size ,spread_weight) #Down
                    draw(event_x,event_y +pixel_size ,spread_weight) #Up
        #Automatically run network
        if draw_thickness > 0:
            if nn.cumalative_int > 30:
                export()
                nn.cumalative_int = 0

            nn.cumalative_int += 1






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
    
    #canvas.bind("<l>", switch_mode)
    

    canvas.focus_set()

    initalise_canvas()
    window.mainloop()


if __name__ == "__main__":
    main()
