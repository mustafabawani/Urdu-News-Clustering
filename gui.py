import tkinter as tk
from tkinter import ttk
import UrduNewsSearch as c

class App(tk.Tk):
    def _init_(self):
        super()._init_()

        # configure the root window
        self.title('Search Engine')
        self.geometry('300x50')
        self.label = tk.Label(self, text="URDU NEWS CLUSTERING USING HEADLINES",bg='#2E8B57', fg='#ffffff', pady=8, padx=8, font=40)
        self.label.pack(pady=10)
        self.label.place(x=320,y=5)
        self.label = tk.Label(self, text="Enter here to Search!",bg='#2E8B57', fg='#ffffff', pady=8, padx=8, font=10)
        self.label.pack()
        self.label.place(x=410,y=70)
        # label
        self.text1 = tk.Text(self,height=1.47,width= 40)
        self.text1.pack()
        self.text1.place(x=340,y=110)        
        # button
        self.button = ttk.Button(self, text='Search')
        self.button['command'] = self.search_button        #Clicking the search button will call the search_button function
        self.button.pack()
        self.button.place(x=680,y=108)
        self.result = tk.StringVar()

    #print docs
        self.result1 = tk.StringVar()
        self.resultShow1 = tk.Label(self, height=23,width= 60, textvariable=self.result1,bg='#ffffff', fg='#2E8B57',pady=8, padx=8, font=("Courier", 10))
        self.resultShow1.place(x=340,y=200)
    #print clusters
        self.result2 = tk.StringVar()
        self.resultShow2 = tk.Label(self, height=30,width= 20, textvariable=self.result2,bg='#ffffff', fg='#2E8B57',pady=8, padx=8, font=("Courier", 10))
        self.resultShow2.place(x=110,y=110)


    def search_button(self):
        input=self.text1.get(1.0,'end-1c')       #Storing user input in input variable
        print(input)
        output = c.clustering_process(input)          #get the resultingposting list
        print(output)
        string=""
        for i in range(len(output)):
            string=string+output[i]+'\n'
        #     string=string+str(output[i])+','

        self.result1.set(string)                 #Safe and Output the resultant list.
        
if __name__ == "__main__":
    app = App()
    app.geometry('1000x600')        
    app.configure(bg='#2E8B57')
    app.resizable(False, False)
    app.mainloop()