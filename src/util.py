
import os
import time
import json
import collections
import argparse, ast
import random
from difflib import SequenceMatcher

from eps import file_list

class Tm:
    """Simple timer for benchmarking"""
    def __init__(self):
        self.cur_time = time.time()
    def do(self, name=None):
        cur_str = '%.3g' % (time.time() - self.cur_time)
        if name: cur_str = ' = '.join([str(name), cur_str])
        self.cur_time = time.time()
        return cur_str

def filter_items(dicts, **filters):
    """Filter list of dicts by criteria"""
    result = dicts
    for key, value in filters.items():
        if value is not None:
            if not isinstance(value, list): value = [value]
            result = [item for item in result if item.get(key) in value]
    return result

def rand_pick(alls, lastpick=None):
    """Random selection avoiding repeats"""
    while True:
        newpick = random.choice(alls)
        if lastpick is None or newpick != lastpick or len(alls) == 1:
            return newpick

def fuzzy_find(items, name, key='name', threshold=0.65):
    """Find item in list of dicts by exact or fuzzy match on a string field"""
    if not name: 
        return None
    name = name.lower()
    for item in items:
        item_val = item.get(key, '')
        if item_val and SequenceMatcher(None, name, item_val.lower()).ratio() > threshold:
            return item
    return None

def max_num(dicts, field):
    """Find max value for a field in list of dicts"""
    return max((item.get(field, 0) for item in dicts), default=0)

def fnexist(fname, ext):
    """Check if files matching pattern exist"""
    workdir = os.path.dirname(fname)
    if not workdir: workdir = './'
    fnames = [f for f in file_list(workdir, ext) if fname in f]
    return fnames

def savelog(txt, dir='./', fname='log.txt'):
    print(txt)
    with open(os.path.join(dir, fname), 'a', encoding="utf-8") as f:
        f.writelines(txt)

def load_args(path, args=None):
    args = args or argparse.Namespace()
    for line in open(path):
        if ":" not in line: continue
        k, v = map(str.strip, line.split(":", 1))
        try:
            v = ast.literal_eval(v)
        except Exception:
            v = None if v == "" else v
        setattr(args, k, v)
    return args

def isok(*itms): # all not None, len > 0
    ok = all([x is not None for x in itms])
    if ok: ok = ok and all([len(x) > 0 for x in itms if hasattr(x, '__len__')])
    return ok

def isset(a, *itms): # all exist, not None, not False, len > 0
    if not all([isinstance(itm, str) for itm in itms]):
        print('!! Wrong items:', *itms); return False
    oks = [True]
    for arg in itms:
        if not hasattr(a, arg) or getattr(a, arg) is None or getattr(a, arg) is False:
            oks += [False]
        elif hasattr(getattr(a, arg), '__len__'):
            oks += [True] if len(getattr(a, arg)) > 0 else [False]
        else:
            oks += [True]
    return all(oks)

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(path, ext=None):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        files = [f for f in files if f.endswith(ext)]
    return sorted([f for f in files if os.path.isfile(f)])

def img_list(path, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    files = [f for f in files if not '/__MACOSX/' in f.replace('\\', '/')] # workaround fix for macos phantom files
    return sorted([f for f in files if os.path.isfile(f)])

def save_cfg(args, dir='./', file='config.txt'):
    if dir != '':
        os.makedirs(dir, exist_ok=True)
    try: args = vars(args)
    except: pass
    if file is None:
        print_dict(args)
    else:
        with open(os.path.join(dir, file), 'w', encoding="utf-8") as cfg_file: # utf-8-sig maybe
            print_dict(args, cfg_file)

def print_dict(dict, file=None, path="", indent=''):
    for k in sorted(dict.keys()):
        if isinstance(dict[k], collections.abc.Mapping):
            if file is None:
                print(indent + str(k))
            else:
                file.write(indent + str(k) + ' \n')
            path = k if path=="" else path + "->" + k
            print_dict(dict[k], file, path, indent + '   ')
        else:
            if file is None:
                print('%s%s: %s' % (indent, str(k), str(dict[k])))
            else:
                file.write('%s%s: %s \n' % (indent, str(k), str(dict[k])))

def txt_clean(txt):
    return ''.join(e for e in txt.replace(' ', '_') if (e.isalnum() or e in ['_','-']))

def backups(out_dir, trees=None, files=None):
    for dest_subpath, src_dir in (trees or {}).items():
        dest_path = os.path.join(out_dir, dest_subpath)
        os.makedirs(dest_path, exist_ok=True)
        shutil.copytree(src_dir, dest_path, dirs_exist_ok=True)
    for dest_subpath, file_list in (files or {}).items():
        dest_path = os.path.join(out_dir, dest_subpath)
        os.makedirs(dest_path, exist_ok=True)
        for f in file_list:
            shutil.copy2(f, os.path.join(dest_path, os.path.basename(f)))
