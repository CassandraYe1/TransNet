import numpy as np
import matplotlib.pyplot as plt
import torch
# from my_utils import grad




class Domain(object):
    def __init__(self, domain_range, domain_shape='box', mesh_domain=None, mesh_boundary=None):
        domain_range = np.array(domain_range)
        domain_range = domain_range.reshape((domain_range.shape[0] // 2, 2))
        self.domain_range = domain_range
        self.domain_dim = domain_range.shape[0]

        # check shape
        self.valid_shapes = ['box', 'annulus', 'circle', 'L-shape']
        if domain_shape in self.valid_shapes:
            pass
        else:
            raise ValueError(f'shape must be in {self.valid_shapes}')

        # different shape only valid in 2d
        if self.domain_dim == 2 or domain_shape == 'box':
            self.domain_shape = domain_shape
        else:
            raise NotImplementedError(f'{domain_shape} not implemented in {self.domain_dim}d')

        if mesh_domain is not None:
            x_train_domain = self.sample_domain_uniform(mesh_size=mesh_domain)
            self.x_train_domain = x_train_domain
        else:
            self.x_train_domain = None

        if mesh_boundary is not None:
            x_train_bd = self.sample_boundary_uniform(sample_size=mesh_boundary)
            self.x_train_bd = x_train_bd
        else:
            self.x_train_bd = None

    def __repr__(self):
        text = f'{self.domain_dim}d {self.domain_shape} domain with range: '
        for i in range(self.domain_dim):
            text += f'{self.domain_range[i]}' if i == 0 else f'*{self.domain_range[i]}'
        return text

    def sample_domain_uniform(self, mesh_size):
        domain_dim = self.domain_dim

        if type(mesh_size) == int:
            mesh_size = [mesh_size] * domain_dim
        elif type(mesh_size) == list and len(mesh_size) == self.domain_dim:
            pass
        else:
            raise ValueError(f'mesh_vec must be list of length {self.domain_dim} or int')

        # generate samples on [-1,1]^n
        x_train_domain_standard = self.sample_nd_mesh(mesh_size)

        if self.domain_shape == 'circle':
            assert self.domain_dim == 2, 'Dim should be 2 for circle.'
            r = np.sqrt(np.sum(x_train_domain_standard ** 2, axis=1))
            index1 = r <= 1.0
            x_train_domain_standard = x_train_domain_standard[index1, :]
            print(x_train_domain_standard.shape)

        if self.domain_shape == 'annulus':
            assert self.domain_dim == 2, 'Dim should be 2 for annulus.'
            r = np.sqrt(np.sum(x_train_domain_standard ** 2, axis=1))
            index1 = r <= 1.0
            index2 = r >= 0.5
            x_train_domain_standard = x_train_domain_standard[index1 * index2, :]

        if self.domain_shape == 'L-shape':
            assert self.domain_dim == 2, 'Dim should be 2 for L-shape.'
            index1 = np.sum(x_train_domain_standard > 0, axis=1) <= 1
            x_train_domain_standard = x_train_domain_standard[index1, :]

        # shift and scale by domain range
        x_train_domain = self.shift2range(x_train_domain_standard)
        return x_train_domain

    def sample_boundary_uniform(self, sample_size):
        if self.domain_shape == 'circle':
            assert self.domain_dim == 2, 'Dim should be 2 for circle.'
            x_train_bd = self.sample_circle_uniform(r=1, sample_size=sample_size)

        if self.domain_shape == 'annulus':
            assert self.domain_dim == 2, 'Dim should be 2 for circle.'
            inner_size = int(sample_size / 3)
            outer_size = sample_size - inner_size
            x_train_boundary_out = self.sample_circle_uniform(r=1, sample_size=outer_size)
            x_train_boundary_in = self.sample_circle_uniform(r=0.5, sample_size=inner_size)
            x_train_bd = np.concatenate([x_train_boundary_out, x_train_boundary_in], axis=0)

        if self.domain_shape == 'L-shape':
            assert self.domain_dim == 2, 'Dim should be 2 for L-shape.'
            sample_1_side = int(np.ceil(sample_size / np.exp2(self.domain_dim))) + 1
            mesh_vec = [sample_1_side] * self.domain_dim
            x_train_bd = self.sample_nd_mesh_bd(mesh_vec=mesh_vec)
            # reshape to L-shape
            x_train_bd = x_train_bd - 1.0 * (x_train_bd == 1) * (np.min(x_train_bd, axis=1,
                                                                        keepdims=True) > 0)

        if self.domain_shape == 'box':
            sample_1_side = sample_size/self.domain_dim/2
            mesh_vec = [int(np.ceil(sample_1_side**(1.0/(self.domain_dim-1))))] * self.domain_dim
            x_train_bd = self.sample_nd_mesh_bd(mesh_vec=mesh_vec)

        # shift and scale by domain range
        x_train_bd = self.shift2range(x_train_bd)
        return x_train_bd

    def shift2range(self, x):
        lb = self.domain_range[:, 0]
        ub = self.domain_range[:, 1]
        x = (x * 0.5 + 0.5) * (ub - lb) + lb
        return x

    @staticmethod
    def sample_circle_uniform(r, sample_size):
        theta_vec = np.linspace(0, 2 * np.pi, sample_size + 1)[:-1]
        x_bd = np.stack([np.cos(theta_vec), np.sin(theta_vec)], axis=1) * r
        return x_bd

    @staticmethod
    def sample_nd_mesh_bd(mesh_vec):
        # generate uniform samples on boundary of [-1,1]^n
        # mesh_size: list of int mesh for each dim
        domain_dim = len(mesh_vec)
        if domain_dim >= 2:
            sample_all = []
            for i in range(domain_dim):
                bd_samples = Domain.sample_nd_mesh(np.delete(mesh_vec, i))
                side1 = np.insert(bd_samples, obj=i, values=1, axis=1)
                side2 = np.insert(bd_samples, obj=i, values=-1, axis=1)
                sample_all.append(side1)
                sample_all.append(side2)
            # generate samples on [-1,1]^n
            box_bd = np.concatenate(sample_all, axis=0)
        else:
            box_bd = np.array([[-1], [1]])
        return box_bd

    @staticmethod
    def sample_nd_mesh(mesh_vec):
        # generate uniform samples on [-1,1]^n
        # mesh_size: list of int mesh for each dim
        domain_dim = len(mesh_vec)
        x_domain_standard = []
        temp_ones = np.ones(mesh_vec)
        for i in range(domain_dim):
            mesh_1d = np.linspace(-1, 1, mesh_vec[i])
            dim_array = np.ones(domain_dim, int)
            dim_array[i] = -1
            mesh_nd = mesh_1d.reshape(dim_array) * temp_ones
            x_domain_standard.append(mesh_nd.flatten())
        x_domain_standard = np.stack(x_domain_standard, axis=1)
        return x_domain_standard


class Problem(object):
    def __init__(self, case=None, data=None):
        self.case = case

        self.x_pde = None
        self.x_bd = None
        self.x_ic = None
        self.x_test = None

        self.target_pde = None
        self.target_bd = None
        self.target_ic = None
        self.target_test = None

        if data is not None:
            self.from_data(data)

        #
        self.pde_name = None
        self.eval_list_pde = None
        self.operator_type = None
        self.eq_names = None
        self.out_var = None

    def from_data(self, data):
        if data is not None:
            if 'x_pde' in data.keys():
                self.x_pde = data['x_pde']

                if len(data['target_pde'].shape) == 0:
                    self.target_pde = data['target_pde'].item()
                else:
                    self.target_pde = data['target_pde']

            if 'x_test' in data.keys():
                self.x_test = data['x_test']
                if len(data['target_test'].shape) == 0:
                    self.target_test = data['target_test'].item()
                else:
                    self.target_test = data['target_test']

            if 'x_bd' in data.keys():
                self.x_bd = data['x_bd']
                if len(data['target_bd'].shape) == 0:
                    self.target_bd = data['target_bd'].item()
                else:
                    self.target_bd = data['target_bd']

            if 'x_ic' in data.keys():
                self.x_ic = data['x_ic']
                if len(data['target_ic'].shape) == 0:
                    self.target_ic = data['target_ic'].item()
                else:
                    self.target_ic = data['target_ic']

    def set_data(self, x_pde,  x_test, target_test, target_pde=None, x_bd=None, x_ic=None, target_bd=None, target_ic=None):
        self.x_pde = x_pde
        self.x_bd = x_bd
        self.x_ic = x_ic
        self.x_test = x_test

        self.target_pde = target_pde
        self.target_bd = target_bd
        self.target_ic = target_ic
        self.target_test = target_test

    def __repr__(self):
        sep = '*****************' * 3
        text_pde = f'{self.pde_name} (case={self.case}):'
        text_train = f'\tx_pde:   \t{None if self.x_pde is None else self.x_pde.shape}'
        text_test = f'\tx_test:    \t{None if self.x_test is None else self.x_test.shape}'
        text_train_bd = f'\tx_bd:    \t{None if self.x_bd is None else self.x_bd.shape}'
        text_train_ic = f'\tx_ic:    \t{None if self.x_ic is None else self.x_ic.shape}'
        return '\n'.join([sep, text_pde, text_train, text_train_bd, text_train_ic, text_test, sep])

    def u_exact(self, x_in):
        # u_exact are used to check solution error and problem
        raise NotImplementedError('Not Implemented')

    def rhs(self, x_in):
        # right hand side of pde (forcing terms)
        raise NotImplementedError('Not Implemented')

    def lhs(self, *args, **kwargs):
        # right hand side of pde (pde operators)
        raise NotImplementedError('Not Implemented')

    def check_solution(self, x_in):
        # right hand side of pde (pde operators)
        raise NotImplementedError('Not Implemented')

    @staticmethod
    def get_grad_auto(u, x):
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:,
              [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:,
               [0]]
        return u_x, u_xx


class Heat_2d_Case1(Problem):
    def __init__(self, case=None, data=None):
        super(Heat_2d_Case1, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Heat_example'
        self.eval_list_pde = ['u', 'u0', 'u1', 'u2', 'u11', 'u22']
        self.eq_names = ['pde']
        self.out_var = 'u'

    def rhs(self, x_in):
        x = x_in[:, [1]]
        return np.zeros_like(x)

    def lhs(self, u_t, u_xx, u_yy):
        # Heat_example operator: u_t - u_xx(t,x)
        return u_t - (u_xx+u_yy)
    
    def initial(self, x_in):
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        y = x_in[:, [2]]
        if type(x) == torch.Tensor:
            u0 = 1e-3 + 100 * torch.exp(-(x**2 + y**2) / 0.01)
        else:
            u0 = 1e-3 + 100 * np.exp(-(x**2 + y**2) / 0.01)
        return u0
    
    def boudary(self, x_in):
        x = x_in[:, [1]]
        return np.zeros_like(x)

    def ls_feature_pde(self, x_in, basis, constrain=True, current_sol=None):
        #assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # Heat_example operator: u_t - u_xx - u_yy
        #x_in = self.x_pde
        feature_pde_all = {}
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)
        dt = 1 / 1000

        feature_pde = basis_eval['u0'] - (basis_eval['u11'] + basis_eval['u22'])
        target_pde = self.rhs(x_in)
        feature_pde_all['pde'] = [feature_pde, target_pde]
        if constrain:
            feature_constrain = basis_eval['u'] - dt * (basis_eval['u11'] + basis_eval['u22'])
            feature_pde_all['constrain'] = [feature_constrain, target_pde]
        
        return feature_pde_all
    
    def ls_feature_boundary(self, x_in, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_bd_all = {}
        l = int(len(x_in) / 4)
        # bd where x = 1
        x_in_right = x_in[:l]
        basis_eval = basis.eval_basis(x_in_right, eval_list=['u1'])
        feature_bd_right = basis_eval['u1']
        target_bd_right = self.boudary(x_in_right)
        feature_bd_all['bd_u_right'] = [feature_bd_right, target_bd_right]
        # bd where x = 0
        x_in_left = x_in[l:2*l]
        basis_eval = basis.eval_basis(x_in_left, eval_list=['u1'])
        feature_bd_left = - basis_eval['u1']
        target_bd_left = self.boudary(x_in_left)
        feature_bd_all['bd_u_left'] = [feature_bd_left, target_bd_left]
        # bd where y = 1
        x_in_top = x_in[2*l:3*l]
        basis_eval = basis.eval_basis(x_in_top, eval_list=['u2'])
        feature_bd_top = basis_eval['u2']
        target_bd_top = self.boudary(x_in_top)
        feature_bd_all['bd_u_top'] = [feature_bd_top, target_bd_top]
        # bd where y = 0
        x_in_bottom = x_in[3*l:]
        basis_eval = basis.eval_basis(x_in_bottom, eval_list=['u2'])
        feature_bd_bottom = - basis_eval['u2']
        target_bd_bottom = self.boudary(x_in_bottom)
        feature_bd_all['bd_u_bottom'] = [feature_bd_bottom, target_bd_bottom]
        
        return feature_bd_all
    
    def ls_feature_initial(self, x_in, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_ic_all = {}
        # ic
        #x_in = self.x_ic
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_ic = basis_eval['u']
        feature_ic_all['ic_u'] = [feature_ic, self.target_ic]

        return feature_ic_all
    

class Heat_2d_Case2(Problem):
    def __init__(self, case=None, data=None):
        super(Heat_2d_Case2, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Heat_example'
        self.eval_list_pde = ['u', 'u0', 'u1', 'u2', 'u11', 'u22']
        self.eq_names = ['pde']
        self.out_var = 'u'

    def rhs(self, x_in):
        x = x_in[:, [1]]
        return np.zeros_like(x)

    def lhs(self, u_t, u_xx, u_yy):
        # Heat_example operator: u_t - u_xx(t,x)
        return u_t - (u_xx+u_yy)
    
    def initial(self, x_in):
        x = x_in[:, [1]]
        return 1e-3 * np.ones_like(x)
    
    def boudary(self, x_in):
        x = x_in[:, [1]]
        return np.ones_like(x)

    def ls_feature_pde(self, x_in, basis, constrain=True, current_sol=None):
        #assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # Heat_example operator: u_t - u_xx - u_yy
        #x_in = self.x_pde
        feature_pde_all = {}
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)
        dt = 1 / 1000

        feature_pde = basis_eval['u0'] - (basis_eval['u11'] + basis_eval['u22'])
        target_pde = self.rhs(x_in)
        feature_pde_all['pde'] = [feature_pde, target_pde]
        if constrain:
            feature_constrain = basis_eval['u'] - dt * (basis_eval['u11'] + basis_eval['u22'])
            feature_pde_all['constrain'] = [feature_constrain, target_pde]
        
        return feature_pde_all
    
    def ls_feature_boundary(self, x_in, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_bd_all = {}
        l = int(len(x_in) / 4)
        # bd where x = 1
        x_in_right = x_in[:l]
        basis_eval = basis.eval_basis(x_in_right, eval_list=['u', 'u1'])
        feature_bd_right = 0.5 * basis_eval['u'] + basis_eval['u1']
        target_bd_right = self.boudary(x_in_right)
        feature_bd_all['bd_u_right'] = [feature_bd_right, target_bd_right]
        # bd where x = 0
        x_in_left = x_in[l:2*l]
        basis_eval = basis.eval_basis(x_in_left, eval_list=['u', 'u1'])
        feature_bd_left = 0.5 * basis_eval['u'] - basis_eval['u1']
        target_bd_left = self.boudary(x_in_left)
        feature_bd_all['bd_u_left'] = [feature_bd_left, target_bd_left]
        # bd where y = 1
        x_in_top = x_in[2*l:3*l]
        basis_eval = basis.eval_basis(x_in_top, eval_list=['u', 'u2'])
        feature_bd_top = 0.5 * basis_eval['u'] + basis_eval['u2']
        target_bd_top = self.boudary(x_in_top)
        feature_bd_all['bd_u_top'] = [feature_bd_top, target_bd_top]
        # bd where y = 0
        x_in_bottom = x_in[3*l:]
        basis_eval = basis.eval_basis(x_in_bottom, eval_list=['u', 'u2'])
        feature_bd_bottom = 0.5 * basis_eval['u'] - basis_eval['u2']
        target_bd_bottom = self.boudary(x_in_bottom)
        feature_bd_all['bd_u_bottom'] = [feature_bd_bottom, target_bd_bottom]
        
        return feature_bd_all
    
    def ls_feature_initial(self, x_in, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_ic_all = {}
        # ic
        #x_in = self.x_ic
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_ic = basis_eval['u']
        feature_ic_all['ic_u'] = [feature_ic, self.target_ic]

        return feature_ic_all


class Heat_Diffusion_1T(Problem):
    def __init__(self, case='z1b1g1Dl', data=None):
        super(Heat_Diffusion_1T, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Heat_diffusion_1T'
        self.eval_list_pde = ['u', 'u0', 'u1', 'u2', 'u11', 'u22']
        self.eq_names = ['pde']
        self.out_var = 'u'

        self.z = 1

    def diff(self, u_old):
        z = self.z
        u_min = 1e-6 * np.ones_like(u_old)
        u_old = np.maximum(u_old, u_min)
        if self.case[7] == 'u': ## Case Dul
            D_1 = u_old**(-1/4) / (4*z**3)
            D_2 = u_old**(3/4) / (3*z**3)
        return D_1, D_2

    def rhs(self, x_in):
        x = x_in[:, [1]]
        return np.zeros_like(x)

    def lhs(self, u_old, u_t, u_x, u_y, u_xx, u_yy):
        # diffusion coefficient: D
        D_1, D_2 = self.diff(u_old)
        return u_t - D_1 * (u_x**2 + u_y**2) - D_2 * (u_xx + u_yy)
        #return u_t - (u**(-1/4)/(4*z**3)) * (u_x**2 + u_y**2) - (u**(3/4)/(3*z**3)) * (u_xx + u_yy)
    
    def initial(self, x_in):
        # initial condition: g
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        y = x_in[:, [2]]
        if self.case[5] == '1': ## Case g1
            return 1e-3 * np.ones_like(x)
        if self.case[5] == '2': ## Case g2
            if type(x) == torch.Tensor:
                u0 = 1e-3 + 100 * torch.exp(-(x**2 + y**2) / 0.01)
            else:
                u0 = 1e-3 + 100 * np.exp(-(x**2 + y**2) / 0.01)
            return u0
    
    def boudary_left(self, x_in):
        # boundary condition on x=0: b
        t = x_in[:, [0]]
        if self.case[3] == '1': ## Case b1
            return np.zeros_like(t)
        if self.case[3] == '2': ## Case b2
            u_bd = 10 * np.ones_like(t)
            for i in range(len(t)):
                if t[i] < 0.5:
                    u_bd[i] = (10/0.5) * t[i]
            return u_bd

    def boudary_others(self, x_in):
        x = x_in[:, [1]]
        return np.zeros_like(x)

    def ls_feature_pde(self, x_in, u_old, basis, constrain=True, current_sol=None):
        #assert feature_name in self.eq_names, 'Invalid pde feature name.'
        feature_pde_all = {}
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)
        dt = 1 / 1000
        
        if current_sol is None:
            D_1, D_2 = self.diff(u_old)
        else:
            u_picard = current_sol['u']
            D_1, D_2 = self.diff(u_picard)
        #feature_pde = basis_eval['u0'] - D_1 * (basis_eval['u1']**2 + basis_eval['u2']**2) - D_2 * (basis_eval['u11'] + basis_eval['u22'])
        feature_pde = basis_eval['u0'] - D_2 * (basis_eval['u11'] + basis_eval['u22'])
        target_pde = self.rhs(x_in)
        feature_pde_all['pde'] = [feature_pde, target_pde]
        if constrain:
            #feature_constrain = basis_eval['u'] - dt * D_1 * (basis_eval['u1']**2 + basis_eval['u2']**2) - dt * D_2 * (basis_eval['u11'] + basis_eval['u22'])
            feature_constrain = basis_eval['u'] - dt * D_2 * (basis_eval['u11'] + basis_eval['u22'])
            feature_pde_all['constrain'] = [feature_constrain, target_pde]
        return feature_pde_all
    
    def ls_feature_boundary(self, x_in, u_old, basis, current_sol=None):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_bd_all = {}
        if current_sol is None:
            D_1, D_2 = self.diff(u_old)
        else:
            u_picard = current_sol['u']
            D_1, D_2 = self.diff(u_picard)
        l = int(len(x_in) / 4)
        # bd where x = 1
        x_in_right = x_in[:l]
        basis_eval = basis.eval_basis(x_in_right, eval_list=['u', 'u1'])
        feature_bd_right = 0.5 * basis_eval['u'] + D_2[-l:] * basis_eval['u1']
        target_bd_right = self.boudary_others(x_in_right)
        feature_bd_all['bd_u_right'] = [feature_bd_right, target_bd_right]
        # bd where x = 0
        x_in_left = x_in[l:2*l]
        basis_eval = basis.eval_basis(x_in_left, eval_list=['u', 'u1'])
        feature_bd_left = 0.5 * basis_eval['u'] - D_2[:l] * basis_eval['u1']
        target_bd_left = self.boudary_left(x_in_left)
        feature_bd_all['bd_u_left'] = [feature_bd_left, target_bd_left]
        # bd where y = 1
        x_in_top = x_in[2*l:3*l]
        basis_eval = basis.eval_basis(x_in_top, eval_list=['u', 'u2'])
        feature_bd_top = 0.5 * basis_eval['u'] + D_2[(l-1)::l] * basis_eval['u2']
        target_bd_top = self.boudary_others(x_in_top)
        feature_bd_all['bd_u_top'] = [feature_bd_top, target_bd_top]
        # bd where y = 0
        x_in_bottom = x_in[3*l:]
        basis_eval = basis.eval_basis(x_in_bottom, eval_list=['u', 'u2'])
        feature_bd_bottom = 0.5 * basis_eval['u'] - D_2[::l] * basis_eval['u2']
        target_bd_bottom = self.boudary_others(x_in_bottom)
        feature_bd_all['bd_u_bottom'] = [feature_bd_bottom, target_bd_bottom]
        
        return feature_bd_all

    def ls_feature_initial(self, x_in, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_ic_all = {}
        # ic
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_ic = basis_eval['u']
        feature_ic_all['ic_u'] = [feature_ic, self.target_ic]

        return feature_ic_all
