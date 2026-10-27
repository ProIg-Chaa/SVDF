"""Fail-closed research file boundary; native opens additionally require strace."""
import json
import os
import sys
import threading
from pathlib import Path


class Boundary:
    def __init__(self, root, generation, output, images, log, public_files=()):
        self.root = os.path.realpath(root)
        self.generation = os.path.realpath(generation)
        self.output = os.path.realpath(output)
        self.images = {os.path.realpath(p) for p in images}
        self.public_files = {os.path.realpath(p) for p in public_files}
        self.model = self.root + '/models/'
        self.libraries = self.root + '/container_python/py310/'
        self.runtime = os.path.join(os.path.dirname(self.output), 'runtime_'+os.environ.get('SLURM_JOB_ID','cpu'))+'/'
        self.fd = os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.counts = dict(opens=0, blocked_opens=0, evaluator_calls=0, blocked_imports=0)
        self.local = threading.local()
        self.local.busy = False

    def record(self, value):
        os.write(self.fd, (json.dumps(value, sort_keys=True)+'\n').encode())

    def audit(self, event, args):
        if getattr(self.local, 'busy', False):
            return
        if event == 'import':
            name = args[0]
            if name in ('lead.data', 'lead.evaluator') or 'canonical_core_evaluator' in name:
                self.counts['blocked_imports'] += 1
                self.record(dict(event='blocked_import', module=name))
                raise PermissionError('Evaluator/data import forbidden during generation: '+name)
        if event != 'open' or isinstance(args[0], int):
            return
        self.local.busy = True
        try:
            path = os.path.realpath(os.fsdecode(args[0]))
            mode, flags = args[1], args[2]
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            reading = (flags & os.O_ACCMODE) != os.O_WRONLY
            within = path == self.root or path.startswith(self.root+'/')
            public = path in self.images or path in self.public_files or path.startswith(self.generation+'/')
            dependency = path.startswith(self.model) or path.startswith(self.libraries)
            mask = path.startswith(self.output+'/masks/') and path.endswith('.png')
            runtime = path.startswith(self.output+'/runtime/') or path.startswith(self.runtime)
            allowed = (not within) or (reading and not writing and (public or dependency or mask))
            if writing and within:
                allowed = (path.startswith(self.output+'/') and (not reading or mask or runtime)) or runtime
            self.counts['opens'] += 1
            self.record(dict(event='open', path=path, mode=mode, flags=flags, reading=reading,
                             writing=writing, allowed=allowed))
            if not allowed:
                self.counts['blocked_opens'] += 1
                raise PermissionError('Generation file boundary: '+path)
        finally:
            self.local.busy = False

    def profile(self, frame, event, arg):
        if event != 'call':
            return
        filename = frame.f_code.co_filename.replace(chr(92), '/')
        if filename.endswith('/evaluator.py') or 'canonical_core_evaluator' in filename:
            self.counts['evaluator_calls'] += 1
            self.record(dict(event='blocked_evaluator_call', file=filename, function=frame.f_code.co_name))
            raise PermissionError('Evaluator call forbidden during generation')

    def install(self):
        sys.addaudithook(self.audit)
        sys.setprofile(self.profile)
        threading.setprofile(self.profile)
        self.record(dict(event='boundary_installed', root=self.root, generation=self.generation,
                         output=self.output, public_images=len(self.images), public_input_files=len(self.public_files), native_monitor='strace required'))

    def snapshot(self):
        value = dict(self.counts)
        value['evaluator_modules_loaded'] = [n for n in sys.modules if n in ('lead.data', 'lead.evaluator') or 'canonical_core_evaluator' in n]
        return value
