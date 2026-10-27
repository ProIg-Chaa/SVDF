"""Keep a native generator frame alive at a pre-token probability callback."""
import queue
import threading
import time


class DiscardTail(Exception):
    pass


class ProbabilityWorker:
    def __init__(self, name, target, snapshot, restore, initial_state):
        self.name=name;self.target=target;self.snapshot=snapshot;self.restore=restore
        self.initial_state=initial_state;self.events=queue.Queue();self.commands=queue.Queue()
        self.status='new';self.error=None;self.result=None;self.payload=None
        self.frame=None;self.state=None;self.pauses=0;self.active_seconds=0.;self.short_active_seconds=0.
        # Healthy workers are joined; a timed-out worker must not keep a failed process alive.
        self.thread=threading.Thread(target=self._run,name=name,daemon=True)

    def pause(self, payload, frame):
        assert self.pauses==0,'Exactly one pre-token pause per active branch'
        self.pauses+=1;self.payload=payload;self.frame=frame;self.state=self.snapshot()
        self.short_active_seconds=time.perf_counter()-self._started
        self.events.put(('paused',None))
        started=time.perf_counter();command=self.commands.get()
        self.active_seconds-=time.perf_counter()-started
        if command=='discard':raise DiscardTail()
        assert command=='resume','Unknown branch command'
        self.restore(self.state)
        self.state=None;self.frame=None

    def _run(self):
        started=time.perf_counter();self._started=started
        try:
            self.restore(self.initial_state);self.initial_state=None
            self.result=self.target(self.pause)
            self.events.put(('complete',None))
        except DiscardTail:
            self.events.put(('discarded',None))
        except BaseException as error:
            self.error=error;self.events.put(('failed',repr(error)))
        finally:
            self.frame=None;self.state=None
            self.active_seconds+=time.perf_counter()-started

    def wait(self, timeout=1800):
        try:status,detail=self.events.get(timeout=timeout)
        except queue.Empty:
            self.status='timed_out'
            self.commands.put('discard')
            raise TimeoutError('Native branch '+self.name+' exceeded '+str(timeout)+' seconds; fatal generation failure')
        self.status=status
        if status in ('complete','discarded','failed'):self.thread.join()
        if status=='failed':raise self.error
        return status

    def start(self):
        assert self.status=='new';self.thread.start();return self.wait()

    def resume(self):
        assert self.status=='paused';self.commands.put('resume');return self.wait()

    def discard(self):
        if self.status=='paused':
            self.commands.put('discard');assert self.wait()=='discarded'
        assert not self.thread.is_alive()
