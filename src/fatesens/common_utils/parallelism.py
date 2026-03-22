# this is a class that takes any function and lists of arguments and runs them in parallel
import multiprocessing
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm_joblib import tqdm_joblib
from typing import Callable, List, Any

class Parallelism:
    def __init__(self, func: Callable, args_list: List[List[Any]], num_workers: int = 0):
        if num_workers <= 0:
            num_workers = multiprocessing.cpu_count()//2
        self.func = func
        self.args_list = args_list
        self.num_workers = num_workers
        
    def run(self) -> List[Any]:
        # print the function name and number of workers
        print(f"Function: {self.func.__name__}, Workers: {self.num_workers}")
        with parallel_backend('loky', n_jobs=self.num_workers):
            print(f"Running in parallel with {self.num_workers}/{multiprocessing.cpu_count()} jobs")
            with tqdm_joblib(desc="Processing", total=len(self.args_list)) as progress_bar:
                results = Parallel(n_jobs=self.num_workers)(delayed(self.func)(*args) for args in self.args_list)
                progress_bar.update(len(results))
        return results
    
def parallelize_function(func: Callable, args_list: List[List[Any]], num_workers: int = 0) -> List[Any]:
    parallelism = Parallelism(func, args_list, num_workers)
    return parallelism.run()