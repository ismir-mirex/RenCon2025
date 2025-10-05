"""
this is a script to render performances from scores based on simples rules


"""
import partitura as pt
import numpy as np
from collections.abc import Callable
import warnings
from numpy.linalg import norm
from scipy.interpolate import CubicSpline

# Filter out specific warnings from partitura
warnings.filterwarnings("ignore", category=UserWarning, module="partitura")
import matplotlib.pyplot as plt


ref_arrays = [
    np.array([[1,0,0,0,1,0,0,1,0,0,0,0], # maj tonic
              [1,0,0,0,1,0,0,1,0,0,0,0]]),
    np.array([[0,0,1,0,0,1,0,0,0,1,0,0],
              [0,0,1,0,0,1,0,0,0,1,0,0]]),
    np.array([[0,0,0,0,1,0,0,1,0,0,0,1],
              [0,0,0,0,1,0,0,1,0,0,0,1]]),
    np.array([[1,0,0,0,0,1,0,0,0,1,0,0],
              [1,0,0,0,0,1,0,0,0,1,0,0]]),
    np.array([[0,0,1,0,0,0,0,1,0,0,0,1], # maj fifth
              [0,0,1,0,0,0,0,1,0,0,0,1]]),
    np.array([[1,0,0,0,1,0,0,0,0,1,0,0],
              [1,0,0,0,1,0,0,0,0,1,0,0]]),
    np.array([[0,0,1,0,0,1,0,0,0,0,0,1],
              [0,0,1,0,0,1,0,0,0,0,0,1]]),
    np.array([[0,0,0,0,1,0,0,0,1,0,0,1], # min dom
              [0,0,0,0,1,0,0,0,1,0,0,1]]),
    np.array([[0,0,1,0,0,0,0,0,1,0,0,1], # min dim
              [0,0,1,0,0,0,0,0,1,0,0,1]]),
    np.array([[0,0,1,0,0,0,1,0,0,1,0,0], # min fourth
              [0,0,1,0,0,0,1,0,0,1,0,0]]),
    np.array([[0,0,0,1,0,0,1,0,0,1,0,0], # min fifth dim
              [0,0,0,1,0,0,1,0,0,1,0,0]]),
    np.array([[0,1,0,0,1,0,0,1,0,0,0,0], # min fourth dim
              [0,1,0,0,1,0,0,1,0,0,0,0]]),
]
ref_triads = [
    set([0,4,7]), # maj tonic
    set([2,5,9]),
    set([4,7,11]),
    set([5,9,0]),
    set([7,11,2]), # maj fifth
    set([9,0,4]),
    set([11,2,5]),
    set([4,8,11]), # min dom
    set([8,11,2]),# min dim
    set([2,6,9]), # min fourth
    set([3,6,9]),# min fifth dim
    set([1,4,7])# min fourth dim
    ] 
        
class ScoreMeasurizer:
    def __init__(self, score_fn):
        self.score_fn = score_fn

    def initialize(self):
        self.score = pt.load_score(self.score_fn)
        self.part = pt.score.unfold_part_maximal(pt.score.merge_parts(self.score.parts), ignore_leaps=False)
        self.note_array = self.part.note_array(include_staff = True)
        
        timepoints, skyline = self.timeseries_skyline(self.note_array)
        self.skyline = skyline
        self.timepoints = timepoints
        self.measures = list()
        self.beat_map = self.part.beat_map
        self.measure_harmony = list()

        for m in self.part.measures:
            print(m)
            ks = self.part.key_signature_map(m.start.t)
            ts = self.part.time_signature_map(m.start.t)
            zero_pitch_to_ks = (ks[0] * 7) % 12
            start = m.start.t
            end = m.end.t
            dur = (end - start) 

            measure_tp = np.arange(self.beat_map(start),self.beat_map(end))
            self.measures.append((measure_tp, ts[0]))

    

            local_na = self.na_within(self.note_array, 
                                    "onset_div", 
                                    start, 
                                    end) # exclusive end
            
            chordbucket = self.chordbucket(local_na,
                                    zero_pitch = zero_pitch_to_ks,
                                    return_array=False)
            self.measure_harmony.append(chordbucket)
              
        return None
    
    def timeseries_skyline(self, 
               note_array, 
               zero_pitch = 0,
               unit="div"):
        unit_to_step = 1
        start = np.min(note_array["onset_"+unit])
        end = np.max(note_array["onset_"+unit] + note_array["duration_"+unit])
        dur = end - start
        
        pianoroll = np.zeros((unit_to_step * dur, 128))
        for n in note_array:
            on = int((n["onset_"+unit] - start) * unit_to_step)
            du = int(n["duration_"+unit]* unit_to_step)
            pianoroll[on: on+du, :(n["pitch"] - zero_pitch)] = 1 

        timepoints = np.unique(note_array["onset_beat"])
        timepoints_unit = np.unique(note_array["onset_"+unit]) * unit_to_step

        sub_pianoroll = pianoroll[timepoints_unit]
        # For each row, find the last column with 1
        skyline_array = np.where(sub_pianoroll == 1, np.arange(sub_pianoroll.shape[1]), 0)
        skyline = skyline_array.max(axis=1)
        return timepoints, skyline

    def chordbucket(self, 
               note_array, 
               zero_pitch = 0,
               return_array = True):

        sm = SetManager()
        for n in note_array:
            sm.add_pitch((n["pitch"] - zero_pitch) % 12)
            idx = sm.check_full_triad()
            if idx is not None:
                return ref_arrays[idx]
        idx = sm.most_likely_triad()
        print("non complete", idx, sm.triads, sm.sizes)
        if return_array:
            return ref_arrays[idx]
        else: 
            return idx

    def na_within(
        self,
        note_array,
        field="onset_beat",
        lower_bound=None,
        upper_bound=None,
        pitch=None,
        exclusion_ids=None,
        inclusion_ids=None,
        ordered_by_field=True,
    ):
        if len(note_array) == 0:
            return list()
        else:
            if pitch is None:
                mask_pitch = np.ones_like(note_array["pitch"])
            else:
                mask_pitch = note_array["pitch"] == pitch

            if lower_bound is None:
                mask_lower = np.ones_like(note_array[field])
            else:
                mask_lower = note_array[field] >= lower_bound

            if upper_bound is None:
                mask_upper = np.ones_like(note_array[field])
            else:
                mask_upper = note_array[field] <= upper_bound

            if exclusion_ids is None:
                mask_exclusion = np.ones_like(note_array[field])
            else:
                mask_exclusion = np.array(
                    [n not in exclusion_ids for n in note_array["id"]]
                )

            if inclusion_ids is None:
                mask_inclusion = np.ones_like(note_array[field])
            else:
                mask_inclusion = np.array([n in exclusion_ids for n in note_array["id"]])

            mask = np.all(
                (mask_pitch, mask_lower, mask_upper, mask_exclusion, mask_inclusion), axis=0
            )
            masked_note_array = note_array[mask]

            if ordered_by_field:
                return masked_note_array[masked_note_array[field].argsort()]
            else:
                return masked_note_array

def normalize_array( arr, a, b):
        arr_min = np.min(arr)
        arr_max = np.max(arr)

        # Avoid division by zero if all values in arr are the same
        if arr_min == arr_max:
            return np.full_like(arr, fill_value=(a + b) / 2)

        # Normalize to [0, 1], then scale to [a, b]
        norm_arr = (arr - arr_min) / (arr_max - arr_min)
        scaled_arr = norm_arr * (b - a) + a
        return scaled_arr

class UpdatableCubicSpline:
    def __init__(self):
        self.points = dict()
        self.spline = lambda x: x
    
    def _update_spline(self):
        # Sort points before constructing spline
        xs, ys = zip(*sorted(self.points.items()))
        self.spline = CubicSpline(xs, ys)
    
    def update_array(self, xs, ys, mode="replace"):
        for x,y in zip(xs, ys):
            self.update_point(x, y, mode=mode, update_spline=False)
        self._update_spline()

    def update_point(self, x, y, mode="replace", update_spline=True):
        """
        Update or insert a point.
        mode = "replace", "add", or "multiply"
        """
        if x in self.points:
            if mode == "replace":
                self.points[x] = y
            elif mode == "add":
                self.points[x] += y
            elif mode == "multiply":
                self.points[x] *= y
            else:
                raise ValueError("Invalid mode. Use 'replace', 'add', or 'multiply'.")
        else:
            if mode == "replace":
                self.points[x] = y
            elif mode == "add":
                self.points[x] = self(x)
                self.points[x] += y
            elif mode == "multiply":
                self.points[x] = self(x)
                self.points[x] *= y
        
        if update_spline:
            self._update_spline()
    
    def __call__(self, x):
        """Evaluate spline at given x values."""
        return self.spline(x)
    
    def get_points(self):
        """Return sorted list of (x, y) points."""
        return sorted(self.points.items())
    
    def get_xs(self):
        return np.array(self.get_points())[:,0]
    
    def get_ys(self):
        return np.array(self.get_points())[:,1]

class SetManager:
    def __init__(self):
        """
        Initialize with reference triads
        """
        self.updates = []  
        self.triads = [
            set([0,4,7]), # maj tonic
            set([2,5,9]),
            set([4,7,11]),
            set([5,9,0]),
            set([7,11,2]), # maj fifth
            set([9,0,4]),
            set([11,2,5]),
            set([4,8,11]), # min fifth
            set([8,11,2]),# min dim
            set([2,6,9]), # min fourth
            set([3,6,9]),# min fifth dim
            set([1,4,7])# min fourth dim
            ] 
        self.sizes = np.full(9,3)
    def add_pitch(self, pitch):
        """
        Add pitch to updates list and remove it from any triad it belongs to.
        """
        if pitch not in self.updates:
            self.updates.append(pitch)
            for s in self.triads:
                if pitch in s:
                    s.remove(pitch)

    def check_full_triad(self):
        """
        Return index of the first empty triad, or False if none are empty.
        """
        for i, s in enumerate(self.triads):
            if len(s) == 0:
                return i
        return None

    def most_likely_triad(self):
        """
        Return index of the triad with the least remaining notes.
        """
        self.sizes = [len(s) for s in self.triads]
        return self.sizes.index(min(self.sizes))

class TempoEvaluator:
    def __init__(self):
        self.curve = UpdatableCubicSpline()
        self.measure_pattern = {
            2: [1.03, 1.0],
            3: [1.03, 1.0, 1.0],
            4: [1.03, 0.99, 1.0, 0.99],
            6: [1.03, 0.99, 0.99, 1.0, 0.9, 0.99]
        }

    def __call__(self, timepoints):
        return self.curve(timepoints)

    def update_curve(self, 
                 timepoints, 
                 tempo,
                 segment_points,
                 measures,
                 skyline):
        self.evaluate_phrases(segment_points)
        for ts in measures:
            self.evaluate_measure(ts[0], ts[1])
        skyline_normalized = normalize_array(skyline, 0.9, 1.1)
        self.curve.update_array(timepoints, skyline_normalized,"multiply")
        self.set_global_tempo(tempo)

    def initialize_one(self, ts):
        self.curve.update_array(ts, np.ones_like(ts),"replace")
    
    def set_global_tempo(self, tempo):
        ts = self.curve.get_xs()
        self.curve.update_array(ts, np.ones_like(ts) * tempo,"multiply")

    def evaluate_measure(self, ts, time_sig):
        pattern = self.measure_pattern[time_sig]
        self.curve.update_array(ts, pattern, "multiply")

    def evaluate_phrases(self, segment_points, high = 1.05 , low = 1/1.05) -> np.ndarray:
        self.curve.update_array(segment_points,
                                 np.ones_like(segment_points)*high,"replace")
        
        mid_points = segment_points[:-1] + np.diff(segment_points) * 2 / 3
        self.curve.update_array(mid_points,
                                 np.ones_like(segment_points)*low,"replace")
        print(segment_points, mid_points)
        # final ritard
        # self.spline.update_point(segment_points[-1], make sure it doesnt go down to early
        #                          1.0,"replace")
        self.curve.update_point(segment_points[-1],
                                 1.8,"replace")
        self.curve.update_point(segment_points[-1] + 4,
                                 2.5,"replace")
        self.curve.update_point(segment_points[0] + 4,
                                 high,"replace")
        
class DynamicsEvaluator:
    def __init__(self):
        self.curve = UpdatableCubicSpline()
        self.measure_pattern = {
            2: [1.03, 1.0],
            3: [1.03, 1.0, 1.0],
            4: [1.03, 0.99, 1.0, 0.99],
            6: [1.03, 0.99, 0.99, 1.0, 0.9, 0.99]
        }

    def __call__(self, timepoints):
        return self.curve(timepoints)

    def update_curve(self, 
                 timepoints, 
                 global_dyn,
                 segment_points,
                 measures,
                 skyline):
        self.evaluate_phrases(segment_points)
        for ts in measures:
            self.evaluate_measure(ts[0], ts[1])
        skyline_normalized = normalize_array(skyline, 0.8, 1.2)
        self.curve.update_array(timepoints, skyline_normalized,"multiply")
        self.set_global_dynamics(global_dyn)
    
    def set_global_dynamics(self, dyn):
        ts = self.curve.get_xs()
        self.curve.update_array(ts, np.ones_like(ts) * dyn,"multiply")

    def evaluate_measure(self, ts, time_sig):
        pattern = self.measure_pattern[time_sig]
        self.curve.update_array(ts, pattern, "multiply")

    def evaluate_phrases(self, segment_points, high = 1.1 , low = 0.9) -> np.ndarray:
        self.curve.update_array(segment_points,
                                 np.ones_like(segment_points)*low,"replace")
        
        
        mid_points = segment_points[:-1] + np.diff(segment_points) * 2 / 3
        self.curve.update_array(mid_points,
                                 np.ones_like(segment_points)*high,"replace")

        self.curve.update_point(segment_points[-1],
                                 0.5,"replace")
        self.curve.update_point(segment_points[-1] + 4,
                                 0.3,"replace")
        self.curve.update_point(segment_points[0] + 4,
                                 low,"replace")
        
class PhraseEvaluator:
    def __init__(self):
            self.phrase_pattern = {
                3: [0.0, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0],
                4: [0.0, 16.0, 32.0, 48.0, 64.0, 80.0,92.0,108.0,144.0],
                6: [0.0, 24.0, 48.0, 72.0, 96.0, 120.0, 156.0, 176.0, 212.0]
            }

    def __call__(self, ts):
        return self.phrase_pattern[ts]


class TimingEvaluator:
    def __init__(self):
        self.timing_dicts = dict()

    def initialize(self, timepoints, note_array):
        for tp in timepoints: 
            mask = note_array["onset_beat"] == tp
            local_pitches = list(note_array[mask]["pitch"])
            timing_dict = dict()
            if len(local_pitches) == 1:
                timing_dict[local_pitches[0]] = 0
            else:
                max_pitch = np.max(local_pitches)
                early = np.random.rand() * -0.01
                timing_dict[max_pitch] = early
                local_pitches.remove(max_pitch)
                splitter = len(local_pitches)
                for remaining_pitch in local_pitches:
                    timing_dict[remaining_pitch] = -1* early / splitter

            self.timing_dicts[tp] = timing_dict

    def __call__(self, ts, pitch):
        return self.timing_dicts[ts][pitch]

class ScoreToPerformanceProcessor:
    def __init__(
        self,
        fn
    ):
        self.scoremeasurizer = ScoreMeasurizer(fn)
        self.scoremeasurizer.initialize()
        self.tempoevaluator = TempoEvaluator()
        self.dynamicsevaluator = DynamicsEvaluator()
        self.timingevaluator = TimingEvaluator()
        self.phraseevaluator = PhraseEvaluator()


    def process(self, tempo = 0.5, global_dyn = 50/128):
        part = self.scoremeasurizer.part
        note_array = self.scoremeasurizer.note_array
        # skyline at beat positions
        skyline = self.scoremeasurizer.skyline
        # beat positions
        timepoints = self.scoremeasurizer.timepoints
        self.timepoints = timepoints
        # [(measure_beats, time_sig)]
        measures = self.scoremeasurizer.measures
        segment_points = self.phraseevaluator(measures[0][1]) # pass time sig
        self.tempoevaluator.update_curve(
                                        timepoints, 
                                        tempo,
                                        segment_points,
                                        measures,
                                        skyline)
        
        self.dynamicsevaluator.update_curve(
                                        timepoints, 
                                        global_dyn,
                                        segment_points,
                                        measures,
                                        skyline)
        
        self.timingevaluator.initialize(timepoints, note_array)
        self.plot_curves(timepoints=timepoints)
        expression_array = self.create_expression_array(note_array)
        
        performance = pt.musicanalysis.decode_performance(part,expression_array)
        pt.save_performance_midi(performance, "render.mid")  
        return performance, expression_array

    def plot_curves(self, timepoints):
        _, ax = plt.subplots(2)
        
        ax[0].plot(timepoints, self.tempoevaluator(timepoints), label='tempo')
        ax[1].plot(timepoints, self.dynamicsevaluator(timepoints), label='dynamics')
        ax[0].legend()
        ax[1].legend()
        name = "curves.png"
        plt.savefig(name)
        plt.close()

    def create_expression_array(self, na):

        fields =[
            ("id", "U256"),
            ("beat_period", "f4"),  # sec/beat
            ("velocity", "f4"), # between 0 and 1
            ("timing", "f4"),  # sec
            ("articulation_log", "f4"),  # log (base 2?) of articulation ratio
        ]
        perf_nl = list()
        last_onset = np.max(na["onset_beat"])
        for note in na:
            perf_tempo = self.tempoevaluator(note["onset_beat"])
            # perf_dyn = np.clip(self.dynamicsevaluator(note["onset_beat"]), 0, 127).astype(int)
            perf_dyn = self.dynamicsevaluator(note["onset_beat"])
            perf_timing = self.timingevaluator(note["onset_beat"], note["pitch"])
            
            if note["onset_beat"] == last_onset:
                perf_art = np.log2(2.0 + np.random.rand()*0.2)
            else:
                perf_art = np.log2(0.6 + np.random.rand()*0.2)

            new_row = (note["id"], perf_tempo, perf_dyn, perf_timing, perf_art)
            perf_nl.append(new_row)

        expression_array = np.array(perf_nl, dtype=fields)
        return expression_array

if __name__=="__main__":

    s2p = ScoreToPerformanceProcessor("path/to/musicxml")
    pe, ea = s2p.process()

