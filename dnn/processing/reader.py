import numpy
import zipfile
import scipy.io

class FileReader(object):
    """
    Read one or multiple numpy arrays from a text/bz2 file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_stream = file_path

    def read_array(self, dtype=numpy.float64, skip_rows=0, use_cols=None, matlab_data_name=None):
        try:
            # Try to read H5:
            if self.file_path.endswith('.h5'):
                return numpy.array([])

            # Try to read NumPy:
            if self.file_path.endswith('.txt') or self.file_path.endswith('.bz2'):
                return self._read_text(self.file_stream, dtype, skip_rows, use_cols)

            if self.file_path.endswith('.npz') or self.file_path.endswith(".npy"):
                return numpy.load(self.file_stream)

            # Try to read Matlab format:
            return self._read_matlab(self.file_stream, matlab_data_name)

        except Exception:
            raise ReaderException("Could not read from %s file" % self.file_path)


    def _read_text(self, file_stream, dtype, skip_rows, use_cols):

        array_result = numpy.loadtxt(file_stream, dtype=dtype, skiprows=skip_rows, usecols=use_cols)
        return array_result


    def _read_matlab(self, file_stream, matlab_data_name=None):

        if self.file_path.endswith(".mtx"):
            return scipy_io.mmread(file_stream)

        if self.file_path.endswith(".mat"):
            matlab_data = scipy_io.matlab.loadmat(file_stream)
            return matlab_data[matlab_data_name]


    def read_gain_from_brainstorm(self):

        if not self.file_path.endswith('.mat'):
            raise ReaderException("Brainstorm format is expected in a Matlab file not %s" % self.file_path)

        mat = scipy_io.loadmat(self.file_stream)
        expected_fields = ['Gain', 'GridLoc', 'GridOrient']

        for field in expected_fields:
            if field not in mat.keys():
                raise ReaderException("Brainstorm format is expecting field %s" % field)

        gain, loc, ori = (mat[field] for field in expected_fields)
        return (gain.reshape((gain.shape[0], -1, 3)) * ori).sum(axis=-1)

class ZipReader(object):
    """
    Read one or many numpy arrays from a ZIP archive.
    """

    def __init__(self, zip_path):
        self.zip_archive = zipfile.ZipFile(zip_path)

    def read_array_from_file(self, file_name, dtype=numpy.float64, skip_rows=0, use_cols=None, matlab_data_name=None):

        matching_file_name = None
        for actual_name in self.zip_archive.namelist():
            if file_name in actual_name and not actual_name.startswith("__MACOSX"):
                matching_file_name = actual_name
                break

        if matching_file_name is None:
            raise ReaderException("File %r not found in ZIP." % file_name)

        zip_entry = self.zip_archive.open(matching_file_name, 'r')

        if matching_file_name.endswith(".bz2"):
            temp_file = copy_zip_entry_into_temp(zip_entry, matching_file_name)
            file_reader = FileReader(temp_file)
            result = file_reader.read_array(dtype, skip_rows, use_cols, matlab_data_name)
            os.remove(temp_file)
            return result

        file_reader = FileReader(matching_file_name)
        file_reader.file_stream = zip_entry
        return file_reader.read_array(dtype, skip_rows, use_cols, matlab_data_name)


    def read_optional_array_from_file(self, file_name, dtype=numpy.float64, skip_rows=0,
                                      use_cols=None, matlab_data_name=None):
        try:
            return self.read_array_from_file(file_name, dtype, skip_rows, use_cols, matlab_data_name)
        except ReaderException:
            return numpy.array([])
