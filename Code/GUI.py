import math
import os
from tkinter import *
from tkinter import ttk, messagebox
import Results
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler


def find_sample_file(sampletype, xi, samplesize, nrsamples, noise_type, **kwargs):
    folder = f'../Data/{sampletype} {xi} xi {samplesize} n {nrsamples} samples/'
    if len(kwargs) == 0:
        # Return without any keywords
        return f'{folder}{sampletype} {xi} xi {samplesize} n {nrsamples} samples {noise_type}.csv'
    if 'param_as_string' in kwargs:
        if kwargs['param_as_string'] != None:
            return f'{folder}{sampletype} {xi} xi {samplesize} n {nrsamples} samples {noise_type} {kwargs["param_as_string"]}.csv'
        else:
            return f'{folder}{sampletype} {xi} xi {samplesize} n {nrsamples} samples {noise_type}.csv'
    dict_string = ''
    for key, value in kwargs.items():
        dict_string += str(key) + str(value)
    return f'{folder}{sampletype} {xi} xi {samplesize} n {nrsamples} samples {noise_type} {dict_string}.csv'


class ResultsInterface:

    def __init__(self, root):
        self.root = root
        self.root.title("Result visualization selection")

        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        # Dict storing which values should be used when pressing the results button
        self.result_dict = {'sample_type': 'RoundedPareto',
                            'xi': 0.5,
                            'sample_size': 100,
                            'nr_samples': 10_000,
                            'noise_type': 'None',
                            'noise_param': None}

        type_selection_frame = ttk.LabelFrame(mainframe, text='Type selection', padding='3 3 12 12')

        # Sample type selection
        self.sample_types_list = ['RoundedPareto', 'MixedPoisson', 'Zipf', 'PreferentialAttachment']
        self.sample_type_selection = ttk.Combobox(type_selection_frame, values=self.sample_types_list, state='readonly', width=21)
        self.sample_type_selection.current(0)
        self.sample_type_selection.bind('<<ComboboxSelected>>', self.sample_type_change)

        # Xi selection
        self.xis_list = [0.5, 0.6, 0.7, 0.8]
        self.xi_selection = ttk.Combobox(type_selection_frame, values=self.xis_list, state='readonly', width=4)
        self.xi_selection.current(0)
        self.xi_selection.bind('<<ComboboxSelected>>', self.xi_change)

        # Sample size selection
        self.sample_sizes_list = [100, 200, 500, 1000, 2000, 5000, 10_000]
        self.sample_size_selection = ttk.Combobox(type_selection_frame, values=self.sample_sizes_list, state='readonly', width=6)
        self.sample_size_selection.current(0)
        self.sample_size_selection.bind('<<ComboboxSelected>>', self.sample_size_change)

        # Noise type selection
        self.noise_types_list = ['None', 'Gaussian', 'Uniform', 'Pareto']
        self.noise_type_selection = ttk.Combobox(type_selection_frame, values=self.noise_types_list, state='readonly', width=9)
        self.noise_type_selection.current(0)
        self.noise_type_selection.bind('<<ComboboxSelected>>', self.noise_type_change)

        # Noise options selection
        self.noise_options = StringVar()
        self.noise_option_selection = ttk.Entry(type_selection_frame, textvariable=self.noise_options, state='disabled')

        ttk.Label(type_selection_frame, text='Sample type').grid(column=0, row=0, sticky='w', padx=5)
        self.sample_type_selection.grid(column=0, row=1, padx=5)
        ttk.Label(type_selection_frame, text='xi').grid(column=1, row=0, sticky='w', padx=5)
        self.xi_selection.grid(column=1, row=1, padx=5)
        ttk.Label(type_selection_frame, text='Sample size').grid(column=2, row=0, sticky='w', padx=5)
        self.sample_size_selection.grid(column=2, row=1, padx=5)
        ttk.Label(type_selection_frame, text='Noise type').grid(column=3, row=0, sticky='w', padx=5)
        self.noise_type_selection.grid(column=3, row=1, padx=5)
        ttk.Label(type_selection_frame, text='Noise type options').grid(column=4, row=0, sticky='w', padx=5)
        self.noise_option_selection.grid(column=4, row=1, padx=5)
        #type_selection_frame.rowconfigure(0, pad='12')

        # Confirmation button
        self.btn_confirm = ttk.Button(mainframe, text='Show results', command=self.show_result_plots, padding='10')
        root.bind('<Return>', lambda e: self.btn_confirm.invoke())
        # When the last plot is closed, the program shuts down. This is an (unintented) interaction between matplotlib and tkinter.
        # Here is a workaround by always having one empty plot in memory
        Results.plot_results_file_list(1, 1, [], (0,0), 10)

        # Results shown selection
        results_selection_frame = ttk.LabelFrame(mainframe, text='Result selection', padding='3 3 12 12')

        self.shown_results_list = []

        self.bool_distance, self.bool_variance, self.bool_mse, \
        self.bool_wald_int, self.bool_score_int, self.bool_lr_int, self.bool_bclr_int, \
        self.bool_wald_rej, self.bool_score_rej, self.bool_lr_rej, self.bool_bclr_rej = \
            BooleanVar(), BooleanVar(), BooleanVar(), \
            BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), \
            BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(),

        self.nr_distance, self.nr_variance, self.nr_mse,\
        self.nr_wald_int, self.nr_score_int, self.nr_lr_int, self.nr_bclr_int,\
        self.nr_wald_rej, self.nr_score_rej, self.nr_lr_rej, self.nr_bclr_rej = \
            IntVar(value=" "), IntVar(value=" "), IntVar(value=" "), \
            IntVar(value=" "), IntVar(value=" "), IntVar(value=" "), IntVar(value=" "), \
            IntVar(value=" "), IntVar(value=" "), IntVar(value=" "), IntVar(value=" ")

        self.result_nr_dict = {'Variance': self.nr_variance, 'Distance': self.nr_distance, 'MSE': self.nr_mse,
                               'WaldIntervals': self.nr_wald_int, 'ScoreIntervals': self.nr_score_int, 'LRIntervals': self.nr_lr_int,
                               'BCLRIntervals': self.nr_bclr_int, 'WaldRejection': self.nr_wald_rej, 'ScoreRejection': self.nr_score_rej,
                               'LRRejection': self.nr_lr_rej, 'BCLRRejection': self.nr_bclr_rej}

        self.lbl_distance = ttk.Label(results_selection_frame, textvariable=self.nr_distance)
        self.lbl_variance = ttk.Label(results_selection_frame, textvariable=self.nr_variance)
        self.lbl_mse = ttk.Label(results_selection_frame, textvariable=self.nr_mse)
        self.lbl_wald_int = ttk.Label(results_selection_frame, textvariable=self.nr_wald_int)
        self.lbl_score_int = ttk.Label(results_selection_frame, textvariable=self.nr_score_int)
        self.lbl_lr_int = ttk.Label(results_selection_frame, textvariable=self.nr_lr_int)
        self.lbl_bclr_int = ttk.Label(results_selection_frame, textvariable=self.nr_bclr_int)
        self.lbl_wald_rej = ttk.Label(results_selection_frame, textvariable=self.nr_wald_rej)
        self.lbl_score_rej = ttk.Label(results_selection_frame, textvariable=self.nr_score_rej)
        self.lbl_lr_rej = ttk.Label(results_selection_frame, textvariable=self.nr_lr_rej)
        self.lbl_bclr_rej = ttk.Label(results_selection_frame, textvariable=self.nr_bclr_rej)
        self.lbl_distance.grid(column=0, row=1, sticky='w')
        self.lbl_variance.grid(column=0, row=2, sticky='w')
        self.lbl_mse.grid(column=0, row=3, sticky='w')
        self.lbl_wald_int.grid(column=0, row=4, sticky='w')
        self.lbl_wald_rej.grid(column=0, row=5, sticky='w')
        self.lbl_score_int.grid(column=0, row=6, sticky='w')
        self.lbl_score_rej.grid(column=0, row=7, sticky='w')
        self.lbl_lr_int.grid(column=0, row=8, sticky='w')
        self.lbl_lr_rej.grid(column=0, row=9, sticky='w')
        self.lbl_bclr_int.grid(column=0, row=10, sticky='w')
        self.lbl_bclr_rej.grid(column=0, row=11, sticky='w')


        self.btn_distance = ttk.Checkbutton(results_selection_frame, text='Distance', command=self.show_res_distance, variable=self.bool_distance)
        self.btn_variance = ttk.Checkbutton(results_selection_frame, text='Variance', command=self.show_res_variance, variable=self.bool_variance)
        self.btn_mse = ttk.Checkbutton(results_selection_frame, text='Mean squared error', command=self.show_res_mse, variable=self.bool_mse)
        self.btn_wald_int = ttk.Checkbutton(results_selection_frame, text='Wald confidence interval size', command=self.show_res_wald_int, variable=self.bool_wald_int)
        self.btn_score_int = ttk.Checkbutton(results_selection_frame, text='Score confidence interval size', command=self.show_res_score_int, variable=self.bool_score_int)
        self.btn_lr_int = ttk.Checkbutton(results_selection_frame, text='LR confidence interval size', command=self.show_res_lr_int, variable=self.bool_lr_int)
        self.btn_bclr_int = ttk.Checkbutton(results_selection_frame, text='BCLR confidence interval size', command=self.show_res_bclr_int, variable=self.bool_bclr_int)
        self.btn_wald_rej = ttk.Checkbutton(results_selection_frame, text='Wald confidence interval rejection probability', command=self.show_res_wald_rej, variable=self.bool_wald_rej)
        self.btn_score_rej = ttk.Checkbutton(results_selection_frame, text='Score confidence interval rejection probability', command=self.show_res_score_rej, variable=self.bool_score_rej)
        self.btn_lr_rej = ttk.Checkbutton(results_selection_frame, text='LR confidence interval rejection probability', command=self.show_res_lr_rej, variable=self.bool_lr_rej)
        self.btn_bclr_rej = ttk.Checkbutton(results_selection_frame, text='BCLR confidence interval rejection probability', command=self.show_res_bclr_rej, variable=self.bool_bclr_rej)
        ttk.Frame(results_selection_frame, width=30, height=0).grid(column=0, row=0)
        self.btn_distance.grid(column=1, row=1, sticky='w')
        self.btn_variance.grid(column=1, row=2, sticky='w')
        self.btn_mse.grid(column=1, row=3, sticky='w')
        self.btn_wald_int.grid(column=1, row=4, sticky='w')
        self.btn_wald_rej.grid(column=1, row=5, sticky='w')
        self.btn_score_int.grid(column=1, row=6, sticky='w')
        self.btn_score_rej.grid(column=1, row=7, sticky='w')
        self.btn_lr_int.grid(column=1, row=8, sticky='w')
        self.btn_lr_rej.grid(column=1, row=9, sticky='w')
        self.btn_bclr_int.grid(column=1, row=10, sticky='w')
        self.btn_bclr_rej.grid(column=1, row=11, sticky='w')


        # Plots configuration
        plot_selection_frame = ttk.LabelFrame(mainframe, text='Plot configuration', padding='3 3 12 12')
        self.nr_rows = IntVar(value=2)
        self.nr_columns = IntVar(value=3)
        self.spinbox_rows = ttk.Spinbox(plot_selection_frame, from_=1, to=11, textvariable=self.nr_rows, state='readonly', command=self.row_change, width=3)
        self.spinbox_columns = ttk.Spinbox(plot_selection_frame, from_=1, to=11, textvariable=self.nr_columns, state='readonly', command=self.column_change, width=3)
        ttk.Label(plot_selection_frame, text='Rows:').grid(row=0, column=0, padx=5, sticky='w')
        self.spinbox_rows.grid(row=0, column=1, pady=5)
        ttk.Label(plot_selection_frame, text='Columns:').grid(row=1, column=0, padx=5, sticky='w')
        self.spinbox_columns.grid(row=1, column=1, pady=5)

        # LabelFrames placement
        noise_options_string = """Uniform noise: loc, scale 
  (0,1), (-1, 1), (-0.5, 1), (-0.5, 0.5), 
  (0, 0.5), (-0.25, 0.5)
Gaussian noise: lower, upper
  (-0.5, 0.5), (0, 1), (-1,0)
Pareto noise: xi
  (0.5), (0.6), (0.7), (0.8)"""
        type_selection_frame.grid(row=0, column=0, columnspan=2, sticky='nswe', padx=(3,12), pady=(3,12))
        results_selection_frame.grid(row=1, column=0, rowspan=3, sticky='nswe', padx=(3,12), pady=(3,12))
        plot_selection_frame.grid(row=2, column=1, padx=(3,12), pady=(3,12), sticky='nswe')
        ttk.Label(mainframe, text=noise_options_string, wraplength=200, anchor='w').grid(row=1, column=1, padx=(3, 12), pady=(3, 12), sticky='nswe')
        self.btn_confirm.grid(row=3, column=1, padx=(3,12), pady=(3,12), sticky='nswe')

        self.root.protocol("WM_DELETE_WINDOW", self.close_main_window)

    def close_main_window(self, *args):
        if messagebox.askokcancel(title='Quit', message='Are you sure you want to quit?'):
            self.root.quit()


    def sample_type_change(self, *args):
        new_sample_type = self.sample_type_selection.get()
        self.result_dict['sample_type'] = new_sample_type
        if new_sample_type == 'PreferentialAttachment':
            self.result_dict['xi'] = 0.5
            self.xi_selection.set(0.5)
            self.xi_selection.configure(state='disabled')
        else:
            self.xi_selection.configure(state='enabled')
            self.xi_selection.configure(state='readonly')


    def xi_change(self, *args):
        self.result_dict['xi'] = float(self.xi_selection.get())

    def sample_size_change(self, *args):
        self.result_dict['sample_size'] = int(self.sample_size_selection.get())

    def noise_type_change(self, *args):
        noise_type = self.noise_type_selection.get()
        self.result_dict['noise_type'] = noise_type
        if noise_type == 'None':
            self.noise_option_selection.configure(state='disabled')
        else:
            self.noise_option_selection.configure(state='enabled')

    def row_change(self, *args):
        self.nr_columns.set(math.ceil(len(self.shown_results_list) / self.nr_rows.get()))

    def column_change(self, *args):
        self.nr_rows.set(math.ceil(len(self.shown_results_list) / self.nr_columns.get()))

    def show_result_plots(self, *args):
        # Get the correct parameters
        if self.result_dict['noise_type'] != 'None':
            self.result_dict['noise_param'] = self.noise_options.get()
        else:
            self.result_dict['noise_param'] = None
        # Find the sample file
        sample_file = find_sample_file(self.result_dict['sample_type'],
                                   self.result_dict['xi'],
                                   self.result_dict['sample_size'],
                                   self.result_dict['nr_samples'],
                                   self.result_dict['noise_type'],
                                   param_as_string=self.result_dict['noise_param'])
        if self.nr_rows.get() * self.nr_columns.get() < len(self.shown_results_list):
            messagebox.showerror(title='Not enough space', message='The number of plots is smaller than the number of selected results.')
            return
        ResultsWindow(self.root, sample_file, self.shown_results_list, self.nr_rows.get(), self.nr_columns.get())

    def show_res_distance(self, *args):
        # The selection is activated
        if self.bool_distance.get():
            self.shown_results_list.append('Distance')
            self.nr_distance.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_distance.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_distance.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_variance(self, *args):
        # The selection is activated
        if self.bool_variance.get():
            self.shown_results_list.append('Variance')
            self.nr_variance.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_variance.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_variance.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_mse(self, *args):
        # The selection is activated
        if self.bool_mse.get():
            self.shown_results_list.append('MSE')
            self.nr_mse.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_mse.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_mse.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_wald_int(self, *args):
        # The selection is activated
        if self.bool_wald_int.get():
            self.shown_results_list.append('WaldIntervals')
            self.nr_wald_int.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_wald_int.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_wald_int.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_score_int(self, *args):
        # The selection is activated
        if self.bool_score_int.get():
            self.shown_results_list.append('ScoreIntervals')
            self.nr_score_int.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_score_int.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_score_int.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_lr_int(self, *args):
        # The selection is activated
        if self.bool_lr_int.get():
            self.shown_results_list.append('LRIntervals')
            self.nr_lr_int.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_lr_int.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_lr_int.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_bclr_int(self, *args):
        # The selection is activated
        if self.bool_bclr_int.get():
            self.shown_results_list.append('BCLRIntervals')
            self.nr_bclr_int.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_bclr_int.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_bclr_int.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_wald_rej(self, *args):
        # The selection is activated
        if self.bool_wald_rej.get():
            self.shown_results_list.append('WaldRejection')
            self.nr_wald_rej.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_wald_rej.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_wald_rej.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_score_rej(self, *args):
        # The selection is activated
        if self.bool_score_rej.get():
            self.shown_results_list.append('ScoreRejection')
            self.nr_score_rej.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_score_rej.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_score_rej.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_lr_rej(self, *args):
        # The selection is activated
        if self.bool_lr_rej.get():
            self.shown_results_list.append('LRRejection')
            self.nr_lr_rej.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_lr_rej.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_lr_rej.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def show_res_bclr_rej(self, *args):
        # The selection is activated
        if self.bool_bclr_rej.get():
            self.shown_results_list.append('BCLRRejection')
            self.nr_bclr_rej.set(len(self.shown_results_list))
        # The selection is deactivated
        else:
            place_in_list = self.nr_bclr_rej.get() - 1
            del self.shown_results_list[place_in_list]
            self.nr_bclr_rej.set(" ")
            # update the rest of the numbers if there are items after the removed item
            if len(self.shown_results_list) > place_in_list:
                 self.update_shown_results()

    def update_shown_results(self):
        for index, result in enumerate(self.shown_results_list):
            self.result_nr_dict[result].set(index + 1)


class ResultsWindow:

    def __init__(self, master, sample_file, measurement_types_list, nr_rows, nr_columns):
        width, height = master.winfo_screenwidth(), master.winfo_screenheight()
        screen_dpi = master.winfo_pixels('1i')
        fig_size = ((width - 0) / screen_dpi, (height - 60) / screen_dpi)
        #
        sample_file_no_ext, _ = os.path.splitext(sample_file)
        measurement_files = [f'{sample_file_no_ext} {type}.csv' for type in measurement_types_list]
        print(measurement_files)
        try:
            self.figure = Results.plot_results_file_list(nr_rows, nr_columns, measurement_files, fig_size, screen_dpi) #, title=os.path.split(sample_file_no_ext)[1])
        except FileNotFoundError:
            messagebox.showerror(title='Results not found', message='The combination of parameters is not correct.')
            return

        self.plot_window = Toplevel(master)
        self.plot_window.state("zoomed")
        self.plot_window.title(os.path.split(sample_file_no_ext)[1])
        mainframe = ttk.Frame(self.plot_window)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        canvas = FigureCanvasTkAgg(self.figure, master=mainframe)
        canvas.get_default_filename = lambda: f'{os.path.split(sample_file_no_ext)[1]}'
        canvas.draw()
        # pack_toolbar=False will make it easier to use a layout manager later on.
        toolbar = NavigationToolbar2Tk(canvas, mainframe, pack_toolbar=False)
        toolbar.update()

        canvas.mpl_connect("key_press_event", key_press_handler)
        toolbar.grid_configure()
        canvas.get_tk_widget().grid_configure()

        self.plot_window.protocol("WM_DELETE_WINDOW", self.close_plot_window)
        self.plot_window.bind("<Control-w>", self.close_plot_window)

    def close_plot_window(self, *args):
        plt.close(self.figure)
        self.plot_window.destroy()

if __name__ == '__main__':
    root = Tk()
    ResultsInterface(root)
    root.mainloop()