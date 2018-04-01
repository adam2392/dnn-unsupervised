class BaseFormat(object):
    '''
    A base formatter for deep neural networks.

    We will be able to make sure that data computations are invariant to the version of python/numpy.
    This can be done by making sure the metadata is saved according to unicode strings always.`
    '''

    def __init__(self):
        pass

    def decodebytes(self, metadata):
        '''
        A method for decoding a metadata dictionary from bytes -> unicode.
        This type of conversion is needed when we have Python2 npz files being saved and then
        read in by the Python3.
        '''
        def convert(data):
            if isinstance(data, bytes):
                return data.decode('ascii')
            if isinstance(data, dict):
                return dict(map(convert, data.items()))
            if isinstance(data, tuple):
                return map(convert, data)
            return data
        # try:
        metadata = {k.decode("utf-8"): (v.decode("utf-8")
                                        if isinstance(v, bytes) else v) for k, v in metadata.items()}
        for key in metadata.keys():
            metadata[key] = convert(metadata[key])
        return metadata

    def loaddatafiles(self, fftdatadir):
    # get all datafiles for the fft maps
    # fftdatadir = '/Volumes/ADAM LI/pydata/output/outputfft/tng/'
    # Get ALL datafiles from all downstream files
        self.datafiles = []
        for root, dirs, files in os.walk(fftdatadir):
            for file in files:
                if '.DS' not in file:
                    self.datafiles.append(os.path.join(root, file))

    def renamefiles(self, project_dir):
        '''
        Function used for renaming the seeg.xyz and gain-inv-square files
        into txt files for easy reading by python.
        '''

        # Initialize files needed to
        # convert seeg.xyz to seeg.txt file
        sensorsfile = os.path.join(project_dir, "seeg.xyz")
        newsensorsfile = os.path.join(project_dir, "seeg.txt")

        try:
            os.rename(sensorsfile, newsensorsfile)
        except:
            print("Already renamed seeg.xyz possibly!")

        # convert gain_inv-square.mat file into gain_inv-square.txt file
        gainmatfile = os.path.join(project_dir, "gain_inv-square.mat")
        newgainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
        try:
            os.rename(gainmatfile, newgainmatfile)
        except:
            print("Already renamed gain_inv-square.mat possibly!")

    def formatdata(self):
        '''
        An abstract function for any child class to inherit and implement
        '''
        raise NotImplementedError(
            'Each formatter for our deep nn needs a formatdata function!')

    def loadseegxyz(self, seegfile):
        '''
        This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
        '''
        seeg_pd = pd.read_csv(
            seegfile, names=['x', 'y', 'z'], delim_whitespace=True)
        self.seegfile = seegfile
        self.seeg_labels = seeg_pd.index.values
        self.seeg_xyz = seeg_pd.as_matrix(columns=None)

    def loadgainmat(self, gainfile):
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(gainfile, header=None, delim_whitespace=True)
        self.gainfile = gainfile
        self.gainmat = gain_pd.as_matrix()

    def loadsurfdata(self, directory, use_subcort=False):
        '''
        Pass in directory for where the entire metadata for this patient is
        '''
        # Shift to account for 0 - unknown region, not included later
        reg_map_cort = np.genfromtxt(
            (os.path.join(directory, "region_mapping_cort.txt")), dtype=int) - 1
        with zipfile.ZipFile(os.path.join(directory, "surface_cort.zip")) as zip:
            with zip.open('vertices.txt') as fhandle:
                verts_cort = np.genfromtxt(fhandle)
            with zip.open('normals.txt') as fhandle:
                normals_cort = np.genfromtxt(fhandle)
            with zip.open('triangles.txt') as fhandle:
                triangles_cort = np.genfromtxt(fhandle, dtype=int)
        vert_areas_cort = self._compute_vertex_areas(
            verts_cort, triangles_cort)

        if use_subcort == False:
            print('NOT USING SUBCORT')
            self.vertices = verts_cort
            self.normals = normals_cort
            self.areas = vert_areas_cort
            self.regmap = reg_map_cort
            return (verts_cort, normals_cort, vert_areas_cort, reg_map_cort)
        else:
            reg_map_subc = np.genfromtxt(
                (os.path.join(directory, "region_mapping_subcort.txt")), dtype=int) - 1
            with zipfile.ZipFile(os.path.join(directory, "surface_subcort.zip")) as zip:
                with zip.open('vertices.txt') as fhandle:
                    verts_subc = np.genfromtxt(fhandle)
                with zip.open('normals.txt') as fhandle:
                    normals_subc = np.genfromtxt(fhandle)
                with zip.open('triangles.txt') as fhandle:
                    triangles_subc = np.genfromtxt(fhandle, dtype=int)
            vert_areas_subc = self._compute_vertex_areas(
                verts_subc, triangles_subc)

            verts = np.concatenate((verts_cort, verts_subc))
            normals = np.concatenate((normals_cort, normals_subc))
            areas = np.concatenate((vert_areas_cort, vert_areas_subc))
            regmap = np.concatenate((reg_map_cort, reg_map_subc))
            self.vertices = verts
            self.normals = normals
            self.areas = areas
            self.regmap = regmap
            return (verts, normals, areas, regmap)

    def __compute_triangle_areas(self, vertices, triangles):
        """Calculates the area of triangles making up a surface."""
        tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
        tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
        tri_norm = np.cross(tri_u, tri_v)
        triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
        triangle_areas = triangle_areas[:, np.newaxis]
        return triangle_areas

    def _compute_vertex_areas(self, vertices, triangles):
        triangle_areas = self.__compute_triangle_areas(vertices, triangles)
        vertex_areas = np.zeros((vertices.shape[0]))
        for triang, vertices in enumerate(triangles):
            for i in range(3):
                vertex_areas[vertices[i]] += 1./3. * triangle_areas[triang]
        return vertex_areas
