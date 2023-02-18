#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import csv
import os
import re
import reprlib
import tempfile
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.io as sio
from IPython.core.error import UsageError
from sos.utils import env, short_repr


def homogeneous_type(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    if first_type in (int, float):
        return True if all(isinstance(x, (int, float)) for x in iseq) else False
    return True if all(isinstance(x, first_type) for x in iseq) else False


scilab_init_statements = r"""
function [repr] = sos_py_repr(obj)
// % isnumeric(A) returns true if A is a numeric array and false otherwise.
// % single Single-precision floating-point array
// % double Double-precision floating-point array
// % int8 8-bit signed integer array
// % uint8 8-bit unsigned integer array
// % int16 16-bit signed integer array
// % uint16 16-bit unsigned integer array
// % int32 32-bit signed integer array
// % uint32 32-bit unsigned integer array
// % int64 64-bit signed integer array
// % uint64 64-bit unsigned integer array
if type(obj) == 1
    // isscalar(A) returns logical 1 (true) if size(A) returns [1 1], and logical 0 (false) otherwise.
    //done
    if size(obj) == [1,1]
        if isinf(obj)
            if obj > 0
                repr = 'np.inf';
            else
                repr = '-np.inf';
            end
        // complex
        elseif isreal(obj) == %f
            rl = string(real(obj));
            im = string(imag(obj));
            repr = strcat(['complex(', rl, ',', im, ')']);
        // none
        elseif isnan(obj)
            repr = 'None';
        else
            repr = sprintf('%f', obj);
        end
    // % isvector(A) returns logical 1 (true) if size(A) returns [1 n] or [n 1] with a nonnegative integer value n, and logical 0 (false) otherwise.
    // DONE!!
    elseif size(obj, 'r')==1 | size(obj, 'c')==1
            if or(isinf(obj))
                repr = strcat(['np.array([', arrfun(obj, sos_py_repr), '])']);
            elseif or(isnan(obj))
                repr = strcat(['np.array([', arrfun(obj, sos_py_repr), '])']);
            elseif and(isreal(obj))
                repr = strcat(['np.array([', arrfun(obj, string), '])']);
            else
                repr = strcat(['np.array([', arrfun(obj, sos_py_repr), '])']);
            end
    // ismatrix(V) returns logical 1 (true) if size(V) returns [m n] with nonnegative integer values m and n, and logical 0 (false) otherwise.
    // DONE!
    elseif size(obj, 'r')>1 && size(obj, 'c')>1 && length(size(obj))==2
        savematfile( TMPDIR + '/mat2py.mat', 'obj', '-v6');
        repr = strcat(['np.matrix(sio.loadmat(r''', TMPDIR, '/mat2py.mat'')', '[''', 'obj', '''])']);
        // outputs: "np.matrix(sio.loadmat(r'/tmp/SCI_TMP_1607379_Bv9sEVmat2py.mat')['obj'])"

    elseif length(size(obj)) >= 3
        //% 3d or even higher matrix
        savematfile( TMPDIR + '/mat2py.mat', 'obj', '-v6');
        repr = strcat(['sio.loadmat(r''', TMPDIR, '/mat2py.mat'')', '[''', 'obj', ''']']);
    // % other, maybe canbe improved with the vector's block
    else
        // % not sure what this could be
        repr = string(obj);
    end


// % char_arr_var
elseif type(obj)==10 && ((size(obj, 'r')>1) ~= (size(obj, 'c')>1))
    repr = '[';
    for i = obj
        repr = strcat([repr, "r''", i, "'',"]);
    end
    repr = part(repr, 1:length(repr)-1);
    repr = strcat([repr,']']);


// % string
// done
elseif type(obj)==10
    repr =strcat(['r""',obj,'""']);
// % structure
// done
elseif isstruct(obj)
    fields = fieldnames(obj);
    repr = '{';
    for i=fields
        repr = strcat([repr, '""', i, '"":', sos_py_repr(obj(i)), ',']);
    end
    repr = strcat([repr, '}']);

    // %save('-v6', fullfile(tempdir, 'stru2py.mat'), 'obj');
    // %repr = strcat('sio.loadmat(r''', tempdir, 'stru2py.mat'')', '[''', 'obj', ''']');

// % cell
//done
elseif iscell(obj)
    if size(obj,1)==1
        repr = '[';
        for i = 1:length(obj)
            repr = strcat([repr, sos_py_repr(obj{i}), ','])
        end
        //done
        repr = strcat([repr,']']);
    else
        //done
        savematfile( TMPDIR + '/cell2py.mat', 'obj', '-v6');
        repr = strcat(['sio.loadmat(r''', TMPDIR, 'cell2py.mat'')', '[''', 'obj', ''']']);
    end
// % boolean
//done
elseif type(obj)==4
    if length(obj)==1
        if obj
            repr = 'True';
        else
            repr = 'False';
        end
    else
        repr = '[';
        for i = obj
            repr = strcat([repr, sos_py_repr(i), ',']);
        end
        repr = strcat([repr,']']);
    end

// % table, table usually is also real, and can be a vector and matrix sometimes, so it needs to be put in front of them.
//DONE!
elseif istable(obj)
    cd (TMPDIR);
    csvWrite(obj,'tab2py.csv',',','QuoteStrings',true);
    repr = strcat(['pd.read_csv(''', TMPDIR, 'tab2py.csv''', ')']);
    else
        // % unrecognized/unsupported datatype is transferred from
        // % matlab to Python as string "Unsupported datatype"
        repr = 'Unsupported datatype';
    end
endfunction


//replace arrayfun
function[newstr] = arrfun(obj, func)
new=[]
for i=obj
    new($+1) = func(i);
    new($+1) = ','
end
new($) = ''
newstr = strcat(new)
endfunction
"""


class sos_scilab:
    supported_kernels = {'Scilab': ['scilab']}
    background_color = '#4bbbff'
    options = {}
    cd_command = 'cd {dir}'

    def __init__(self, sos_kernel, kernel_name='scilab'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = scilab_init_statements

    def _scilab_repr(self, obj, name=None):
        #  Converting a Python object to a scilab expression that will be executed
        #  by the scilab kernel.
        if isinstance(obj, bool):
            return r'%t' if obj else r'%f'
        if isinstance(obj, (int, float, str, complex)):
            if isinstance(obj, complex):
                if obj.imag > 0:
                    return repr(obj.real) + "+" + repr(obj.imag) + r'*%i'
                return repr(obj.real) + repr(obj.imag) + r'*%i'

            return repr(obj)
        if isinstance(obj, Sequence):
            if len(obj) == 0:
                return '[]'
            # if the data is of homogeneous type, let us use []

            if homogeneous_type(obj):
                return '[' + ' '.join(self._scilab_repr(x) for x in obj) + ']'
            return 'list(' + ' '.join(self._scilab_repr(x) for x in obj) + ')'
        if obj is None:
            return r'%nan'
        if isinstance(obj, dict):
            dic = tempfile.tempdir
            os.chdir(dic)
            # change how this is saved to be compatible with scilab
            sio.savemat('dict2mtlb.mat', {'sos_obj__': obj})
            return 'getfield(loadmatfile(fullfile(' + '\'' + dic + '\'' + ',' \
                + '\'dict2mtlb.mat\')), \'obj\'); ' + name + '=sos_obj__'

        # scilab has no sets, closest equivalent is a list
        if isinstance(obj, set):
            return 'list(' + ','.join(self._scilab_repr(x) for x in obj) + ')'
        if isinstance(obj, (
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
        )):
            return repr(obj)

        if isinstance(obj, np.matrixlib.defmatrix.matrix):
            dic = tempfile.tempdir
            os.chdir(dic)

            #need struct2cell alternative cell2mat(struct2cell(
            sio.savemat('mat2mtlb.mat', {'sos_obj__': obj})
            return 'loadmatfile(fullfile(' + '\'' + dic + '\'' + ',' \
                + '\'mat2mtlb.mat\')); ' + name + '=sos_obj__'
        if isinstance(obj, np.ndarray):
            dic = tempfile.tempdir
            os.chdir(dic)

            sio.savemat('ary2mtlb.mat', {'sos_obj__': obj})
            return 'loadmatfile(fullfile(' + '\'' + dic + '\'' + ',' \
                + '\'ary2mtlb.mat\')); ' + name + '=sos_obj__'
        if isinstance(obj, pd.DataFrame):
            if self.kernel_name == 'scilab':
                dic = tempfile.tempdir
                os.chdir(dic)
                obj.to_csv('df2oct.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
                return 'csvRead(' + '\'' + dic + '/' + 'df2oct.csv\')'

            dic = tempfile.tempdir
            os.chdir(dic)
            obj.to_csv('df2mtlb.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
            return 'readtable(' + '\'' + dic + '/' + 'df2mtlb.csv\')'

    async def get_vars(self, names, as_var=None):
        for name in names:
            # add 'm' to any variable beginning with '_'
            if as_var:
                newname = as_var
            elif name.startswith('_'):
                self.sos_kernel.warn(f"Variable {name} is passed from SoS to kernel {self.kernel_name} as {'m' + name}")
                newname = 'm' + name
            else:
                newname = name
            scilab_repr = self._scilab_repr(env.sos_dict[name], name=newname)
            env.log_to_file('KERNEL', f'Executing \n{scilab_repr}')
            await self.sos_kernel.run_cell(
                f'{newname} = {scilab_repr}',
                True,
                False,
                on_error=f'Failed to get variable {name} of type {env.sos_dict[name].__class__.__name__} to scilab')

    def put_vars(self, items, to_kernel=None, as_var=None):
        if not items:
            return {}

        result = {}
        for item in items:
            py_repr = f'sos_py_repr({item})'

            #9 scilab can use multiple messages for standard output,
            # so we need to concatenate these outputs.
            expr = ''
            for _, msg in self.sos_kernel.get_response(py_repr, ('stream',), name=('stdout',)):
                expr += msg['text']

            try:
                if 'loadmat' in expr:
                    # imported to be used by eval
                    from scipy.io import loadmat
                # evaluate as raw string to correctly handle \\ etc
                # expr = expr[expr.index('\n  ') + 4:expr.rindex('\r\n')-3]
                expr = re.sub(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]', '', expr)
                expr = re.sub(r'[\b\r\n]*', '', expr).split('=', 1)[-1].strip()
                if not expr.startswith('"') or not expr.endswith('"'):
                    raise ValueError(f'Invalid return expresion "{expr}"')
                result[as_var if as_var else item] = eval(expr[1:-1])
            except Exception as e:
                self.sos_kernel.warn(f'Failed to evaluate {expr!r}:\n {e}')
        return result

    def sessioninfo(self):
        return self.sos_kernel.get_response(r'ver', ('stream',), name=('stdout',))[0][1]['text']
