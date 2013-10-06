import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import wave

from worlds.base_world import World as BaseWorld
import core.tools as tools
import becca_tools_control_panel.control_panel as cp

class World(BaseWorld):
    """ The listen world provides a stream of audio data to the BECCA agent
    There are no actions that the agent can take that affect the world. 
    """
    # This package assumes that it is located directly under the BECCA package 
    def __init__(self, lifespan=None):
        super(World, self).__init__()
        if lifespan is not None:
            self.LIFESPAN = lifespan
        # Flag indicates whether the world is in testing mode
        self.TEST = False
        self.DISPLAY_INTERVAL = 10 ** 4
        self.plot_all_features = True
        self.name = 'listen_world'
        print "Entering", self.name
        self.sample_length_ms = 200
        pad_length_ms = 1000
        self.SAMPLING_FREQUENCY = 44100.
        self.pad_length = int(np.floor(pad_length_ms * 
                                       self.SAMPLING_FREQUENCY / 1000)) 
        self.snippet_length = int(np.floor(self.sample_length_ms * 
                                           self.SAMPLING_FREQUENCY / 1000))
        # Step through the data such that each time step is synchronized with
        # a frame of video
        frames_per_time_step = 3
        self.audio_samples_per_time_step = (
                frames_per_time_step * int(self.SAMPLING_FREQUENCY / 29.947))
        # Generate a list of the filenames to be used
        if self.TEST:
            self.audio_filenames = []
            filename = os.path.join('becca_world_listen', 'test', 'test.txt')
            self.audio_filenames.append(filename)
            self.ground_truth_filename = os.path.join('becca_world_listen', 
                                                      'test', 'truth.txt')
        else:
            self.data_dir_name = os.path.join('becca_world_listen', 'data')
            extensions = ['.txt']
            self.audio_filenames = tools.get_files_with_suffix(
                    self.data_dir_name, extensions)
        self.audio_file_count = len(self.audio_filenames)
        print self.audio_file_count, 'audio files loaded.'
        # Initialize the image_data to be viewed
        self.initialize_audio_file()

        self.frequencies = np.fft.fftfreq(self.snippet_length, 
                                d = 1/self.SAMPLING_FREQUENCY) 
        self.keeper_frequency_indices = np.where(self.frequencies > 0)
        self.frequencies = self.frequencies[self.keeper_frequency_indices] 
        tones_per_octave = 12.
        min_log2_freq = 5.
        num_octaves = 8.
        max_log2_freq = min_log2_freq + num_octaves
        num_bin_boundaries = num_octaves * tones_per_octave + 1
        self.bin_boundaries = np.logspace(min_log2_freq, max_log2_freq, 
                             num=num_bin_boundaries, endpoint=True, base=2.)
        self.bin_boundaries = np.concatenate((
                np.ones(1) * tools.EPSILON, self.bin_boundaries, 
                np.ones(1) * (np.amax(self.frequencies) + tools.EPSILON)))
        bin_membership = np.digitize(self.frequencies, self.bin_boundaries)
        self.bin_map = np.zeros((self.bin_boundaries.size - 1, 
                                self.frequencies.size))
        for bin_map_row in range(self.bin_map.shape[0]):
            self.bin_map[bin_map_row, 
                         np.where(bin_membership-1 == bin_map_row)] = 1.
        self.bin_map = self.bin_map / (np.sum(self.bin_map, axis=1) 
                                       [:,np.newaxis] + tools.EPSILON)
        # Hann window 
        self.window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(
                        self.snippet_length) / (self.snippet_length - 1)))
        self.num_sensors = self.bin_map.shape[0]
        self.num_actions = 0
        self.initialize_control_panel()
        self.frame_counter = 10000
        self.last_feature_visualized = 0
        if self.TEST:
            self.surprise_log_filename = os.path.join('becca_world_listen', 
                                                      'log', 'surprise.txt')
            self.surprise_log = open(self.surprise_log_filename, 'w')

    def initialize_audio_file(self):
        filename = self.audio_filenames[np.random.randint(
                0, self.audio_file_count)]
        print 'Loading', filename
        self.audio_data = np.loadtxt(filename)
        self.audio_data = np.delete(self.audio_data, 
                                    np.where(np.isnan(self.audio_data)), 0)
        self.padded_audio_data = np.concatenate((
                np.zeros(self.pad_length), 
                self.audio_data, np.zeros(self.pad_length)))
        self.position_in_clip = 0
        if self.audio_data.size < self.snippet_length:
            print 'That clip was too short. Trying another.'
            self.initialize_audio_file()

    def step(self, action): 
        self.timestep += 1
        # Check whether the end of the clip has been reached
        if (self.position_in_clip + 5 * self.snippet_length > 
            self.audio_data.size):
            if self.TEST:
                # Terminate the test
                self.surprise_log.close()
                print 'End of test reached'
                tools.report_roc(self.ground_truth_filename, 
                                self.surprise_log_filename, self.name)
                sys.exit()
            else:
                self.initialize_audio_file()
        # Generate a new audio snippet and set of sensor values
        self.snippet = self.audio_data[self.position_in_clip: 
                self.position_in_clip + self.snippet_length] * self.window

        magnitudes = np.abs(np.fft.fft(self.snippet).real) \
                            [self.keeper_frequency_indices]
        #self.sensors = np.dot(self.bin_map, magnitudes[:,np.newaxis])
        binned_magnitudes = np.dot(self.bin_map, magnitudes[:,np.newaxis])
        self.sensors = np.log2(binned_magnitudes + 1.)
        reward = 0
        self.position_in_clip += self.audio_samples_per_time_step 
        return self.sensors, reward
        
    def set_agent_parameters(self, agent):
        agent.VISUALIZE_PERIOD = 10 ** 3
        # debug
        #agent.recent_surprise_history = [21.] * 100
        #agent.typical_surprise = 21.
        #agent.filtered_surprise = agent.typical_surprise
        if self.TEST:
            # Prevent the agent from adapting during testing
            agent.BACKUP_PERIOD = 10 ** 9
            for block in agent.blocks:
                block.ziptie.COACTIVITY_UPDATE_RATE = 0.
                block.ziptie.JOINING_THRESHOLD = 2.
                block.ziptie.AGGLOMERATION_ENERGY_RATE = 0.
                block.ziptie.NUCLEATION_ENERGY_RATE = 0.
                for cog in block.cogs:
                    cog.ziptie.COACTIVITY_UPDATE_RATE = 0.
                    cog.ziptie.JOINING_THRESHOLD = 2.
                    cog.ziptie.AGGLOMERATION_ENERGY_RATE = 0.
                    cog.ziptie.NUCLEATION_ENERGY_RATE = 0.
                    cog.daisychain.CHAIN_UPDATE_RATE = 0.
        else:
            pass
    
    def initialize_control_panel(self):
        self.fig = cp.figure()
        self.ax_snippet_long = cp.subfigure(self.fig, 
                left=0., bottom=0.65, width=0.6, height=0.35)
        self.ax_snippet = cp.subfigure(self.fig, 
                left=0., bottom=0.35, width=0.6, height=0.3)
        self.ax_status = cp.subfigure(self.fig, 
                left=0., bottom=0., width=0.2, height=0.35)
        self.ax_sensors = cp.subfigure(self.fig, 
                left=0.2, bottom=0., width=0.4, height=0.35)
        #self.ax23 = cp.subfigure(self.fig, 
        #        left=0.6, bottom=0., width=0.25, height=0.1)

        # Initialize long snippet and surprise plot 
        self.long_snippet_length = self.pad_length * 2
        t_max = self.long_snippet_length * 1000 / self.SAMPLING_FREQUENCY
        self.time_steps_per_long_snippet = int(self.long_snippet_length / \
                self.audio_samples_per_time_step) 
        t_steps = np.linspace(0, t_max, self.time_steps_per_long_snippet)
        t = np.linspace(0, t_max, self.long_snippet_length)
        self.snippet_data_long, = self.ax_snippet_long.plot(t, 
                np.zeros((self.long_snippet_length)), color=tools.COPPER_SHADOW)
        min_x_limit = 0.
        max_x_limit = t_max
        self.ax_snippet_long.axis((min_x_limit, max_x_limit, -1., 1.))
        self.ax_snippet_long.add_patch(mpatches.Rectangle((900, -.99), 200, 
                1.98, facecolor=tools.LIGHT_COPPER, 
                edgecolor=tools.COPPER_SHADOW) )
        self.ax_snippet_long.text(min_x_limit +
                (max_x_limit - min_x_limit) * 0.05, 0.97, 'Audio data stream', 
                color=tools.COPPER_SHADOW, size=10, ha='left', va='bottom')
        self.ax_snippet_long.set_xlabel('time (ms)      .', 
                color=tools.COPPER_SHADOW, size=10, ha='right', va='center')
        self.ax_snippet_long.hold(True)
        self.surprise_data, = self.ax_snippet_long.plot(t_steps, -1 * np.ones((
                self.time_steps_per_long_snippet)), color=tools.COPPER,
                linewidth=2.)
        self.ax_snippet_long.text(min_x_limit +
                (max_x_limit - min_x_limit) * 0.05, -.97, 'Novelty', 
                color=tools.COPPER, size=10, ha='left', va='bottom')
        
        # Initialize snippet plot 
        t_max = self.snippet_length * 1000 / self.SAMPLING_FREQUENCY
        t = np.linspace(0, t_max, self.snippet_length)
        self.snippet_data, = self.ax_snippet.plot(t, 
                np.zeros((self.snippet_length)), color=tools.COPPER_SHADOW)
        min_x_limit = 0.
        max_x_limit = t_max
        self.ax_snippet.axis((min_x_limit, max_x_limit, -1., 1.))
        self.ax_snippet.text(min_x_limit + (max_x_limit - min_x_limit) * 0.05, 
                0.97, 'Audio snippet', color=tools.COPPER_SHADOW, size=10,
                ha='left', va='bottom')
        self.ax_snippet.set_xlabel('time (ms)      .', 
                color=tools.COPPER_SHADOW, 
                size=10, ha='right', va='center')
        
        # Initialize sensors window 
        self.bar_width = 0.01
        self.sensor_data = self.ax_sensors.barh(
                np.log10(self.bin_boundaries[1:]) - self.bar_width/2, 
                np.zeros(self.num_sensors), height=self.bar_width, 
                color=tools.COPPER_SHADOW)
        self.min_y_sensor_limit = np.log10(30)
        self.max_y_sensor_limit = np.log10(20000)
        self.ax_sensors.axis((0., 5., self.min_y_sensor_limit, 
                self.max_y_sensor_limit))
        self.fig.canvas.draw()
        labels = [item.get_text() for item 
                  in self.ax_sensors.get_yticklabels()]
        labels[0] = ''
        labels[1] = ''
        labels[2] = '100'
        labels[3] = ''
        labels[4] = '1000'
        labels[5] = ''
        labels[6] = '10000'

        self.ax_sensors.set_yticklabels(labels)
        self.ax_sensors.text(0.05, self.min_y_sensor_limit + 
                (self.max_y_sensor_limit - self.min_y_sensor_limit) * 0.94, 
                'Sensors', color=tools.COPPER_SHADOW, size=10, 
                ha='left', va='bottom')
        self.ax_sensors.set_ylabel('frequency (Hz)', color=tools.COPPER_SHADOW, 
                size=10, ha='right', va='center')
    
        # Initialize status window 
        self.ax_status.axis((0., 1., 0., 1.))
        self.ax_status.get_xaxis().set_visible(False)
        self.ax_status.get_yaxis().set_visible(False)
        self.clip_time_status = self.ax_status.text(-0.05, 0.9,
                    'Clip time:', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')
        self.wake_time_status = self.ax_status.text(-0.05, 0.8,
                    'Wake time:', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')
        self.life_time_status = self.ax_status.text(-0.05, 0.7,
                    'Life time:', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')
        
        # Initialize surprise plot 
        self.surprise_ax_left = 0.6
        self.surprise_ax_bottom = 0.56
        self.surprise_ax_width = 0.4
        self.surprise_ax_height = 0.44
        self.ax_surprise = cp.subfigure(self.fig, left=self.surprise_ax_left, 
                            bottom=self.surprise_ax_bottom, 
                            width=self.surprise_ax_width, 
                            height=self.surprise_ax_height)
        self.ax_surprise.axis((0., 1., 0., 1.))
        self.ax_surprise.get_xaxis().set_visible(False)
        self.ax_surprise.get_yaxis().set_visible(False)
        self.block_ax_vert_border = 0.02 * self.surprise_ax_height
        self.block_ax_horz_border = 0.04 * self.surprise_ax_width
        self.surprise_block_ax = []
        
        # Initialize features plot 
        self.feature_ax_left = 0.6
        self.feature_ax_bottom = 0.12
        self.feature_ax_width = 0.4
        self.feature_ax_height = 0.44
        self.ax_features = cp.subfigure(self.fig, left=self.feature_ax_left, 
                            bottom=self.feature_ax_bottom, 
                            width=self.feature_ax_width, 
                            height=self.feature_ax_height)
        self.ax_features.axis((0., 1., 0., 1.))
        self.ax_features.get_xaxis().set_visible(False)
        self.ax_features.get_yaxis().set_visible(False)
        self.feature_ax_vert_border = 0.025 * self.feature_ax_height
        self.feature_ax_horz_border = 0.005 * self.feature_ax_width
        self.block_ax = []
        
        # Initialize heartbeat plot         
        #self.x = np.linspace(0, 6*np.pi, 100)
        #self.line1, = self.ax23.plot(self.x, np.sin(self.x), 'k-')
        #self.ax23.get_xaxis().set_visible(False)
        #self.ax23.get_yaxis().set_visible(False)
        #self.phase = 0.
        self.fig.show()

    def visualize(self, agent):
        if self.TEST:
            # Save the surprise value
            surprise_val = agent.surprise_history[-1]
            time_in_seconds = str(float(self.position_in_clip) / 
                                  float(self.SAMPLING_FREQUENCY))
            file_line = ' '.join([str(surprise_val), str(time_in_seconds)])
            self.surprise_log.write(file_line)
            self.surprise_log.write('\n')

        if (self.timestep % self.DISPLAY_INTERVAL != 0):
            return 
        print self.timestep, 'steps'
        # Update surprise data 
        half_length = int(self.time_steps_per_long_snippet/ 2)
        surprise = [0] * self.time_steps_per_long_snippet
        if len(agent.surprise_history) < half_length:
            surprise[half_length - len(agent.surprise_history): half_length] \
                    = agent.surprise_history
        else:
            surprise[:half_length] = agent.surprise_history[-half_length:]
        surprise_mod = np.abs(np.asarray(surprise) / 30.) - 1. 
        surprise_mod = np.minimum(surprise_mod, 1.)
        self.surprise_data.set_ydata(surprise_mod)
        # Update long snippet data
        start_sample = self.position_in_clip - self.pad_length * .9 + \
                                self.pad_length            
        long_snippet = self.padded_audio_data[start_sample: 
                                    start_sample + self.long_snippet_length]
        scale_factor = 0.5
        long_snippet = long_snippet / scale_factor 
        self.snippet_data_long.set_ydata(long_snippet)
        # Update snippet plot 
        self.snippet_data.set_ydata(self.snippet)
        # Update sensors window 
        for i in range(len(self.sensor_data)):
            self.sensor_data[i].set_width(self.sensors[i])
        # Update status window 
        self.clip_time_status.set_text(''.join((
                'Clip time: ', '%0.2f' % (self.position_in_clip /
                (self.SAMPLING_FREQUENCY * 60.)), ' min')))
        self.wake_time_status.set_text(''.join((
                'Wake time: ', '%0.2f' % (self.timestep *
                self.audio_samples_per_time_step /
                (self.SAMPLING_FREQUENCY * 60.)), ' min')))
        self.life_time_status.set_text(''.join((
                'Life time: ', '%0.2f' % (agent.timestep *
                self.audio_samples_per_time_step /
                (self.SAMPLING_FREQUENCY * 60.)), ' min')))

        # Update surprise visualization window
        # Clear all axes
        for axes in self.surprise_block_ax:
            self.fig.delaxes(axes)
        self.surprise_block_ax = []
        # Display each block's features and bundle activities.
        # The top block has no bundles.
        num_blocks = len(agent.blocks)
        for block_index in range(num_blocks):
            block = agent.blocks[block_index]
            block_surprise = block.surprise / 10.
            num_cogs_in_block = len(block.cogs)
            surprise_array = np.reshape(block_surprise, 
                                        (num_cogs_in_block,
                                         block.max_bundles_per_cog)).T
            block_left = self.surprise_ax_left + self.block_ax_horz_border 
            block_height = ((self.surprise_ax_height -
                             self.block_ax_vert_border - 
                             self.feature_ax_vert_border * 2) / 
                             float(num_blocks) - 
                            self.block_ax_vert_border)
            block_bottom = (self.surprise_ax_bottom + 
                            self.feature_ax_vert_border +
                            self.block_ax_vert_border +
                            (block_height + self.block_ax_vert_border) * 
                            block_index)
            block_width = self.surprise_ax_width - \
                            2 * self.block_ax_horz_border
            last_block_top = block_bottom + block_height
            rect = (block_left, block_bottom, block_width, block_height)
            ax = self.fig.add_axes(rect, frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.gray()
            im = ax.imshow(surprise_array, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=1.,
                           cmap='copper')
            if block_index == 0:
                ax.text(num_cogs_in_block * 0.85,
                        block.max_bundles_per_cog * 0.8, 
                        'Novelty', color=tools.OXIDE, 
                        size=10, ha='left', va='bottom')
            self.surprise_block_ax.append(ax)

        # Update feature visualization window
        # Clear all axes
        for axes in self.block_ax:
            self.fig.delaxes(axes)
        self.block_ax = []
        (projections, feature_activities) = agent.get_projections()
        # Display each block's features and bundle activities.
        # The top block has no bundles.
        num_blocks = len(agent.blocks)
        for block_index in range(num_blocks):
            block = agent.blocks[block_index]
            cable_activities = block.cable_activities
            num_cogs_in_block = len(block.cogs)
            activity_array = np.reshape(cable_activities, 
                                        (num_cogs_in_block,
                                         block.max_bundles_per_cog)).T
            block_left = self.feature_ax_left + self.block_ax_horz_border 
            block_height = ((self.feature_ax_height -
                             self.block_ax_vert_border - 
                             self.feature_ax_vert_border * 2) / 
                             float(num_blocks) - 
                            self.block_ax_vert_border)
            block_bottom = (self.feature_ax_bottom + 
                            self.feature_ax_vert_border +
                            self.block_ax_vert_border +
                            (block_height + self.block_ax_vert_border) * 
                            block_index)
            block_width = self.feature_ax_width - \
                            2 * self.block_ax_horz_border
            last_block_top = block_bottom + block_height
            rect = (block_left, block_bottom, block_width, block_height)
            ax = self.fig.add_axes(rect, frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.gray()
            im = ax.imshow(activity_array, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=1.,
                           cmap='copper')
            if block_index == 0:
                ax.text(num_cogs_in_block * 0.85,
                        block.max_bundles_per_cog * 0.8, 
                        'Activities', color=tools.OXIDE, 
                        size=10, ha='left', va='bottom')
            self.block_ax.append(ax)
        for block_index in range(num_blocks - 1):
            if self.plot_all_features:
                for feature_index in range(len(projections[block_index])):
                    states_per_feature = block_index + 2
                    plt.close(99)
                    feature_fig = plt.figure(num=99)
                    for state in range(states_per_feature):                
                        left =  float(state) / float(states_per_feature) 
                        bottom = 0.
                        width =  1. /  float(states_per_feature)
                        height =  1
                        rect = (left, bottom, width, height)
                        ax = feature_fig.add_axes(rect)
                        bar_centers = np.log10(
                                self.bin_boundaries[1:]) - \
                                self.bar_width/2 
                        bar_centers[-1] += self.bar_width
                        bar_color = tools.COPPER_SHADOW
                        ax.barh(bar_centers,  
                                projections[block_index][feature_index]
                                [self.num_actions:self.num_actions + 
                                 self.num_sensors, state], 
                                height=self.bar_width, 
                                color=bar_color, edgecolor=bar_color)
                        ax.plot(np.asarray((0.03, 0., 0., 0.03)), 
                                np.asarray((2.05, 2.05, 2.95, 2.95)), 
                                color=tools.COPPER_SHADOW, linewidth=2)
                        ax.axis((0., 1., self.min_y_sensor_limit, 
                                 self.max_y_sensor_limit))
                    # create a plot of individual features
                    filename = '_'.join(('block', str(block_index).zfill(2),
                                         'feature',str(feature_index).zfill(4),
                                         self.name, 'world.png'))
                    full_filename = os.path.join('becca_world_listen',
                                                 'log', filename)
                    plt.title(filename)
                    plt.savefig(full_filename, format='png') 
                    # Create an audio representation of the feature
                    audio_state_duration = 0.3 # seconds
                    audio_state_overlap = 0.2# seconds
                    audio_feature_duration = (audio_state_duration + 
                            (audio_state_duration - audio_state_overlap) * 
                            (states_per_feature - 1))
                    audio_state_length = int(audio_state_duration * 
                                          self.SAMPLING_FREQUENCY)
                    audio_feature_length = int(audio_feature_duration * 
                                            self.SAMPLING_FREQUENCY)
                    audio_feature = np.zeros(audio_feature_length)
                    bin_centers = self.bin_boundaries[:-1] + np.diff(
                            self.bin_boundaries) 
                    hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(
                            audio_state_length) / (audio_state_length - 1)))
                    for state in range(states_per_feature):                
                        audio_state = np.zeros(audio_state_length)
                        state_bins = (projections[block_index][feature_index]
                                [self.num_actions:self.num_actions + 
                                self.num_sensors, state]) 
                        start_index = int(state * (audio_state_duration -
                                       audio_state_overlap) * 
                                       self.SAMPLING_FREQUENCY)
                        end_index = start_index + audio_state_length 
                        for state_bin_index in np.nonzero(state_bins)[0]:
                            omega = bin_centers[state_bin_index] * 2 * 3.14159
                            phase = np.random.random_sample() * 2 * 3.14159
                            t = (np.cumsum(np.ones(audio_state.shape)) / 
                                 self.SAMPLING_FREQUENCY)   
                            bin_tone = np.sin(omega * t + phase)
                            windowed_bin_tone = bin_tone * hann_window
                            audio_state += windowed_bin_tone
                        audio_feature[start_index:end_index] += audio_state
                    # Scale audio data
                    audio_feature *= (0.8 * tools.MAX_INT16 / 
                                   np.max(np.abs(audio_feature)))
                    audio_feature = audio_feature.astype(np.int16)
                    filename = '_'.join(('block', str(block_index).zfill(2),
                                         'feature',str(feature_index).zfill(4),
                                         'listen', 'world.wav'))
                    feature_audio_filename = os.path.join('becca_world_listen',
                                                 'log', filename)
                    wave_file = wave.open(feature_audio_filename, 'wb')
                    nchannels = 1
                    sampwidth = 2
                    framerate = int(self.SAMPLING_FREQUENCY)
                    nframes = audio_feature.size
                    comptype = "NONE"
                    compname = "not compressed"
                    wave_file.setparams((nchannels, sampwidth, framerate, 
                                         nframes, comptype, compname))
                    # I have no idea why it only works when I write it twice.
                    # If I only write it once, I only get the first half
                    # of the file.
                    wave_file.writeframes(audio_feature)
                    wave_file.writeframes(audio_feature)
                    wave_file.close()
        self.fig.canvas.draw()
        plt.draw()
        # Save the control panel image
        filename =  self.name + '_' + str(self.frame_counter) + '.png'
        full_filename = os.path.join('becca_world_listen', 'frames', filename)
        self.frame_counter += 1
        plt.figure(self.fig.number)
        plt.savefig(full_filename, format='png', dpi=80) # for 720
        #plt.savefig(full_filename, format='png', dpi=120) # for 1080
        return
