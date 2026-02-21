from __future__ import annotations
from matplotlib.dates import SA
import numpy as np
import scipy.stats as stats
from numpy.random import default_rng
import openturns as ot
import matplotlib.pyplot as plt
from math import comb


# Global registry 
workflow_name_to_class_registry={}
sampler_name_to_class_registry={}
distribution_name_to_class_registry={}
post_workflow_name_to_class_registry={}

#Orchestrator class, manages workflows and interfaces with uam4x

def register_workflow(Workflow):
    workflow_name_to_class_registry[Workflow.__name__]=Workflow
    return Workflow

def register_sampler(Sampler):
    sampler_name_to_class_registry[Sampler.__name__]=Sampler
    return Sampler

def register_distribution(Distribution):
    distribution_name_to_class_registry[Distribution.__name__]=Distribution
    return Distribution 

def register_post_workflow(PostWorkflow):
    post_workflow_name_to_class_registry[PostWorkflow.__name__]=PostWorkflow
    return PostWorkflow



class UQAnalysis:
    #MIGHT CHANGE THIS TO ALEATORYANALYSIS, 
    #THEN MAYBE HAVE A MIXED ANALYSIS 
    # AND PURE EPISTEMIC ALL ESSENTIALLY AT PARAMETERSWEEP/OPTIMIZATION level
    def __init__(self,UQ_workflow,seed=None):
        self.UQ_workflow = UQ_workflow
        self.seed=seed if seed is not None else 42        
        self.aleatory_var_to_dist = {}
        self.post_spec=pre_to_post_workflow_registry[UQ_workflow.__class__.__name__]


    def _check_analysis(self):
        if hasattr(self.UQ_workflow,'_check_analysis'):
            self.UQ_workflow._check_analysis(self)
    

    #def set_post_class(self,PostClass):
    #    self.post_spec=PostClass

    def add_aleatory_variable(self, var_name, PDFObject):
        self.aleatory_var_to_dist[var_name] = PDFObject

    def get_states(self):
        ot.RandomGenerator.SetSeed(self.seed)
        #self.UQ_workflow.
        return self.UQ_workflow._get_states(self.aleatory_var_to_dist)
    
    def _to_dict(self):
        return {
            'type':self.__class__.__name__,
            'seed':self.seed,
            'aleatory_variables':{name:dist._to_dict() for name, dist in self.aleatory_var_to_dist.items()},
            'analysis_workflow':self.UQ_workflow._to_dict(),
            'post_processing':self.post_spec.__name__ if self.post_spec else None
        }
    
    @classmethod
    def _from_dict(cls,UQdict):
        
        #load in the workflow from dictionary
        workflow_dict=UQdict['analysis_workflow']
        workflow=workflow_name_to_class_registry[workflow_dict['name']]._from_dict(workflow_dict)
        # load in seed
        seed=UQdict['seed']
        #instantiate class
        UQanalysis=cls(workflow,seed)
        #iterate through the aleatory variables to readd
        for name,distribution_data in UQdict['aleatory_variables'].items():
            dist=distribution_name_to_class_registry[distribution_data['name']]._from_dict(distribution_data)
            UQanalysis.add_aleatory_variable(name,dist)
        if UQdict['post_processing'] is not None:
            UQanalysis.post_spec=post_workflow_name_to_class_registry[UQdict['post_processing']]
        return UQanalysis

        

#Analysis workflow pre classes
@register_workflow
class BasicStats:

    #post_class=BasicStatsPost

    def __init__(self,num_samples,SamplingClass):
        self.num_samples = num_samples
        self.SamplingClass = SamplingClass

    def _get_states(self,aleatory_to_distribution):
        #states should be a list of dictionaries
        sample_points,var_name_to_col=self.SamplingClass.sample(aleatory_to_distribution,self.num_samples)
        states=[]
        for row in range(sample_points.shape[0]):
            local_state={}
            for name,col in var_name_to_col.items():
                local_state[name]=sample_points[row,col]
            local_state['optimization']=False
            states.append(local_state)
        return states
    
    def _to_dict(self):
        return {
            'name':self.__class__.__name__,
            'sampler': self.SamplingClass.__name__,
            'num_samples':self.num_samples
        }
    
    @classmethod
    def _from_dict(cls,workflow_dict):
        sampler=sampler_name_to_class_registry[workflow_dict['sampler']]
        num_samples=workflow_dict['num_samples']
        return cls(num_samples,sampler)
    
@register_workflow
class RegressionPCE(BasicStats):
    def __init__(self,num_samples,SamplingClass,poly_order):
        super().__init__(num_samples,SamplingClass)
        self.poly_order=poly_order

    def _check_analysis(self,UQAnalysis):
        #check order and number of aleatory vs. num samples
        # from XIU Polynomial expansion, dimension of space is C(order+number of random vars,number of random vars)
        # should oversample 2-3x
        num_aleatory=len(UQAnalysis.aleatory_var_to_dist)
        min_terms=comb(self.poly_order+num_aleatory,self.poly_order)
        if min_terms>self.num_samples:
            raise ValueError(f'Insufficient terms for polynomial chaos expansion of degree {self.poly_order} for {num_aleatory} random variables')
        elif min_terms*3>self.num_samples:
            print('Warning')
        
    def _to_dict(self):
        return {
            'name': self.__class__.__name__,
            'sampler':self.SamplingClass.__name__,
            'num_samples':self.num_samples,
            'polynomial_order':self.poly_order
        }
    
    @classmethod
    def _from_dict(cls,workflow_dict):
        sampler=sampler_name_to_class_registry[workflow_dict['sampler']]
        num_samples=workflow_dict['num_samples']
        poly_order=workflow_dict['polynomial_order']
        return cls(num_samples,sampler,poly_order)


@register_workflow
class GaussQuadraturePCE:
    def __init__(self,poly_order):
        self.poly_order=poly_order

    def _check_analysis(self,UQAnalysis):
        num_aleatory=len(UQAnalysis.aleatory_var_to_dist)
        if num_aleatory>5:
            raise ValueError('Quadrature based PCE may not be efficient for more than 5 aleatory variables')

    def _get_states(self, aleatory_to_distribution):
        num_aleatory=len(aleatory_to_distribution)

        if num_aleatory==1:
            (name,dist),=aleatory_to_distribution.items()
            var_name_to_col={name:0}
            dist=dist._OT_Dist()
        else:
            joint=[]
            var_name_to_col={}
            for i,(name, dist) in enumerate(aleatory_to_distribution.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name]=i
            dist=ot.JointDistribution(joint)
            
        # Experiment with joint distribution with k marginals
        # the points per var is poly deg+1, the total number of points is
        # (deg+1)^k for full tensor implementation (used here) 
        points_per_var=[self.poly_order+1]*num_aleatory
        experiment=ot.GaussProductExperiment(dist,points_per_var)
        nodes,weights=experiment.generateWithWeights()
        nodes,weights=np.array(nodes),np.array(weights)

        states=[]
        for row in range(nodes.shape[0]):
            local_state={'weight':weights[row]}
            for name,col in var_name_to_col.items():
                local_state[name]=nodes[row,col]
            local_state['optimization']=False
            states.append(local_state)

        return states
                


    def _to_dict(self):
        return {
            'name': self.__class__.__name__,
            'polynomial_order':self.poly_order
        }
    
    @classmethod
    def _from_dict(cls,workflow_dict):
        poly_order=workflow_dict['polynomial_order']
        return cls(poly_order)

@register_workflow
class SobolSensitivity(BasicStats):
    '''
    Docstring for SobolSensitivity
    '''
    def __init__(self,num_samples,SamplingClass,computeSecondOrder:bool=False):
        super().__init__(num_samples,SamplingClass)
        self.computeSecondOrder=computeSecondOrder
    
    def _check_analysis(self):
        pass

    def _get_states(self, aleatory_to_distribution):
        ot.ResourceMap.Set('SobolIndicesExperiment-SamplingMethod',self.SamplingClass._OT_ResourceMapName)
        if len(aleatory_to_distribution)==1:
            (name,dist),=aleatory_to_distribution.items()
            var_name_to_col={name:0}
            dist=dist._OT_Dist()
        else:
            joint=[]
            var_name_to_col={}
            for i,(name, dist) in enumerate(aleatory_to_distribution.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name]=i
            dist=ot.JointDistribution(joint)
        
        experiment=ot.SobolIndicesExperiment(dist,self.num_samples,self.computeSecondOrder)
        results=np.array(experiment.generate())

        # check matrix size: 
        # from OpenTurns documentation: if computeSecondOrder is false, number of rows is N(2+nx) where N is num
        # samples and nx is number of marginals (distributions/aleatory vars)
        # if computeSecondOrder is true and nx=2, same as above
        # elif computeSecondOrder and nx!=2, rows is N(2+2*nx)
        if (not self.computeSecondOrder) or len(aleatory_to_distribution)==2:
            if results.shape[0]!=self.num_samples*(2+len(aleatory_to_distribution)):
                raise ValueError(f'Number of rows is unexpected {results.shape[0]}')
        else:
            if results.shape[0]!=self.num_samples*(2+2*len(aleatory_to_distribution)):
                raise ValueError(f'Number of rows is unexpected {results.shape[0]}')

        states=[]
        for row in range(results.shape[0]):
            local_state={}
            for name,col in var_name_to_col.items():
                local_state[name]=results[row,col]
                local_state['optimization']=False
            states.append(local_state)
            

        return states
    
    
    def _to_dict(self):
        return {
            'name': self.__class__.__name__,
            'sampler':self.SamplingClass.__name__,
            'num_samples':self.num_samples,
            'second_order_sobol':self.computeSecondOrder,
            'num_sobol_samples':self.num_sobol_samples
        }
    
    @classmethod
    def _from_dict(cls,workflow_dict):
        sampler=sampler_name_to_class_registry[workflow_dict['sampler']]
        num_samples=workflow_dict['num_samples']
        second_order_sobol=workflow_dict['second_order_sobol']
        num_sobol_samples=workflow_dict['num_sobol_samples']

        sobol_workflow=cls(num_samples,sampler,second_order_sobol)
        sobol_workflow.num_sobol_samples=num_sobol_samples

        return sobol_workflow
    

# Define post process classes:
@register_post_workflow
class BasicStatsPost:
    def __init__(self,tradestudy:Tradestudy):
            self.tradestudy=tradestudy

    def compute_mean(self,var_name):
        vals=self.tradestudy.get_variable(var_name)
        return np.average(vals)

    def compute_variance(self, var_name,method='sample'):        
        vals = self.tradestudy.get_variable(var_name)
        sample=['s','sample']
        population=['p','pop','population']
        if method in sample:
            return float(np.var(vals, ddof=1))
        elif method in population:
            return float(np.var(vals, ddof=0))
        else:
            raise ValueError()
                 
    def compute_std(self, var_name,method='sample'):
        vals = self.tradestudy.get_variable(var_name)
        sample=['s','sample']
        population=['p','pop','population']
        if method in sample:
            return float(np.std(vals, ddof=1))
        elif method in population:
            return float(np.std(vals, ddof=0))
        else:
            raise ValueError()
        

    def compute_skewness(self,var_name,bias=True):

        # uses Fisher-Pearson
        vals = np.array(self.tradestudy.get_variable(var_name))
        return stats.skew(vals,nan_policy='raise',bias=bias)

    def compute_kurtosis(self,var_name,fisher=True,bias=True):
        # uses fisher or pearson
        vals = np.array(self.tradestudy.get_variable(var_name))
        return stats.kurtosis(vals,nan_policy='raise',bias=bias,fisher=fisher)

    def plot_pdf(self,var_name,x_axis_title=None,plot_title='',method='histogram',bins=20,save_dir=None):
        histo=['histogram','h']
        kernal=['kde','kernel_density_estimate','kernel','k']
        if method in histo:
            vals = np.array(self.tradestudy.get_variable(var_name))
            fig,ax=plt.subplots(figsize=(10,6))
            ax.hist(vals,bins=bins,color='blue',edgecolor='black')
            ax.set_title(plot_title if plot_title else f'PDF of {var_name}', fontsize=16,fontweight='bold')
            ax.set_xlabel(x_axis_title if x_axis_title is not None else var_name, fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
        elif method in kernal:
            vals = np.array(self.tradestudy.get_variable(var_name))
            fig,ax=plt.subplots(figsize=(10,6))
            kde = stats.gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(x_range, kde(x_range), color='blue')
            ax.set_title(plot_title if plot_title else f'PDF of {var_name}', fontsize=16,fontweight='bold')
            ax.set_xlabel(x_axis_title if x_axis_title is not None else var_name, fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
        else:
            raise ValueError()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/PDF_{var_name.replace(' ','_')}.png", dpi=300, bbox_inches='tight')

        return ax
        
    def plot_cdf(self,var_name,x_axis_title=None,plot_title='',save_dir=None):
        vals=np.array(self.tradestudy.get_variable(var_name))
        ecdf=stats.ecdf(vals)
        quantiles=ecdf.cdf.quantiles
        probabilities=ecdf.cdf.probabilities
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(quantiles,probabilities,color='blue')
        if x_axis_title is None:
            x_axis_title=var_name    
        if plot_title=='':
            plot_title=f'CDF of {x_axis_title}'
        ax.set_title(plot_title, fontsize=16,fontweight='bold')
        ax.set_xlabel(x_axis_title, fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/{plot_title.replace(' ','_')}.png", dpi=300, bbox_inches='tight')  

        return np.hstack((quantiles,probabilities))

    def get_cdf_function(self,var_name,method='empirical'):
        vals=np.array(self.tradestudy.get_variable(var_name))
        ecdf=stats.ecdf(vals)
        quantiles=ecdf.cdf.quantiles
        probabilities=ecdf.cdf.probabilities
        if method=='empirical':
            def cdf(x):
                return ecdf.cdf.evaluate(x) 
        elif method=='smooth':
            def cdf(x):
                return np.interp(x,quantiles,probabilities,left=0.0,right=1.0)
        else:
            raise ValueError
        return cdf
    
    def compute_cdf_value(self,var_name,value,method='empirical'):
        vals=np.array(self.tradestudy.get_variable(var_name))
        ecdf=stats.ecdf(vals)
        quantiles=ecdf.cdf.quantiles
        probabilities=ecdf.cdf.probabilities
        if method=='empirical':
            return ecdf.cdf.evaluate(value) 
        elif method=='smooth':
            return np.interp(value,quantiles,probabilities,left=0.0,right=1.0)
        else:
            raise ValueError

    
    def compute_quantile(self,var_name,quantile,method='inverted_cdf'):
        ''''
        Computes probability levels (quantiles)
        '''
        if quantile<=0.0 or quantile>=1.0:
            raise ValueError()
        methods=[
            'inverted_cdf',
            'averaged_inverted_cdf',
            'closest_observation',
            'interpolated_inverted_cdf',
            'hazen',
            'weibull',
            'linear',
            'median_unbiased',
            'normal_unbiased'
        ]

        if method not in methods:
            raise ValueError
        
        vals=np.array(self.tradestudy.get_variable(var_name))

        return np.quantile(vals,quantile,method=method)                

    def compute_spearman_correlation(self,var_name1,var_name2):
        #NEED TO DETERMINE NAN POLICY
        var1_vals=np.array(self.tradestudy.get_variable(var_name1))
        var2_vals=np.array(self.tradestudy.get_variable(var_name2))
        significance_res,p_val=stats.spearmanr(var1_vals,var2_vals)
        #SHOULD HANDLE >3 VARS REQUIRES MATRIX
        return significance_res,p_val

    def compute_pearson_correlation(self,var_name1,var_name2):
        #NEED TO DETERMINE NAN POLICY
        var1_vals=np.array(self.tradestudy.get_variable(var_name1))
        var2_vals=np.array(self.tradestudy.get_variable(var_name2))
        significance_res,p_val=stats.pearsonr(var1_vals,var2_vals)
        #SHOULD HANDLE >3 VARS REQUIRES MATRIX
        return significance_res,p_val



class MixedBasicStatsPost(BasicStatsPost):
    def __init__(self,tradestudy:Tradestudy):
        super().__init__(tradestudy)

    def plot_probability_box():
        pass

    def compute_confidence_intervals():
        pass



@register_post_workflow
class RegressionPCEPost(BasicStatsPost):
    def __init__(self,tradestudy:Tradestudy):
        super().__init__(tradestudy)

    def generate_surrogate(self,output_name):
        trade_name=self.tradestudy.trade_name
        ts=self.tradestudy.study.control.trade_studies[trade_name]
        num_samples=ts.UQ_workflow.num_samples
        aleatory_var_to_dist=ts.aleatory_var_to_dist
        poly_factories=[]
        abscissas=np.zeros((num_samples,len(aleatory_var_to_dist)))
        joint_list=[]
        for i,(name,distribution) in enumerate(aleatory_var_to_dist.items()):
            ot_dist=distribution._OT_Dist()
            poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
            abscissas[:,i]=np.array(self.tradestudy.get_variable(name))
            joint_list.append(ot_dist)
        
        # Data as OT samples
        abscissas=ot.Sample(abscissas)
        outputs=ot.Sample(np.array(self.tradestudy.get_variable(output_name))[:,None])

        # Joint distribution based on independent input aleatory dists
        joint=ot.JointDistribution(joint_list)

        #Enumeration function
        enum=ot.LinearEnumerateFunction(len(aleatory_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis=ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order=ts.UQ_workflow.poly_order
        basis_size=enum.getBasisSizeFromTotalDegree(poly_order)
        if basis_size!=comb(poly_order+len(aleatory_var_to_dist),poly_order):
            raise ValueError(f'Unexpected basis size {basis_size} for poly order {poly_order} and num aleatory {len(aleatory_var_to_dist)}')
            
        # Selects first P elemens of the polynomial chaos basis according to enumeration function
        truncation_strategy=ot.FixedStrategy(product_basis,basis_size)

        # This is regression based PCE, so projection is done via least squares minimization
        projection_strategy=ot.LeastSquaresStrategy()

        algo=ot.FunctionalChaosAlgorithm(abscissas,outputs,joint,truncation_strategy,projection_strategy)
        algo.run()
        res=algo.getResult()

        input_order=list(ts.aleatory_var_to_dist.keys())

        return Surrogate(input_order,res)
    
@register_post_workflow    
class GaussQuadtraturePCEPost(BasicStatsPost):
    def __init__(self,tradestudy:Tradestudy):
        super().__init__(tradestudy)

    def generate_surrogate(self,output_name):
        trade_name=self.tradestudy.trade_name
        ts=self.tradestudy.study.control.trade_studies[trade_name]
        aleatory_var_to_dist=ts.aleatory_var_to_dist
        num_aleatory=len(aleatory_var_to_dist)
        num_samples=(ts.UQ_workflow.poly_order+1)**num_aleatory
        #get the weights:
        weights=[]
        for result in self.tradestudy._results:
            weights.append(result.state['weight'])
        weights=ot.Point(weights)
        poly_factories=[]
        # matrix of input values
        abscissas=np.zeros((num_samples,len(aleatory_var_to_dist)))
        joint_list=[]
        for i,(name,distribution) in enumerate(aleatory_var_to_dist.items()):
            ot_dist=distribution._OT_Dist()
            poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
            abscissas[:,i]=np.array(self.tradestudy.get_variable(name))
            joint_list.append(ot_dist)

        # Data as OT samples
        abscissas=ot.Sample(abscissas)
        outputs=ot.Sample(np.array(self.tradestudy.get_variable(output_name))[:,None])

        # Joint distribution based on independent input aleatory dists
        joint=ot.JointDistribution(joint_list)


        #Enumeration function
        enum=ot.LinearEnumerateFunction(len(aleatory_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis=ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order=ts.UQ_workflow.poly_order
        basis_size=enum.getBasisSizeFromTotalDegree(poly_order)
        if basis_size!=comb(poly_order+len(aleatory_var_to_dist),poly_order):
            raise ValueError(f'Unexpected basis size {basis_size} for poly order {poly_order} and num aleatory {len(aleatory_var_to_dist)}')
            
        # Selects first P elemens of the polynomial chaos basis according to enumeration function
        truncation_strategy=ot.FixedStrategy(product_basis,basis_size)

        # Gaussian PCE using integration 
        projection_strategy=ot.IntegrationStrategy()

        algo=ot.FunctionalChaosAlgorithm(abscissas,weights,outputs,joint,truncation_strategy,projection_strategy)
        algo.run()
        res=algo.getResult()

        input_order=list(ts.aleatory_var_to_dist.keys())

        return Surrogate(input_order,res)

class Surrogate:
    def __init__(self,inp_order,result):
        self.input_order=inp_order
        self._ot_res=result
        self.new_sample_points=None
        self.new_sample_values=None
        
    
    def predict(self,name_to_value_dict):
        vals=[]
        for name in self.input_order:
            vals.append(name_to_value_dict[name])
        point=ot.Point(vals)
        return self._ot_res.getMetaModel()(point)
    
    @property
    def analytical_mean(self):
        return ot.FunctionalChaosRandomVector(self._ot_res).getMean()[0]
    @property
    def analytical_variance(self):
        ot_var=np.array(ot.FunctionalChaosRandomVector(self._ot_res).getCovariance())[0][0]
        coeffs=np.array(self._ot_res.getCoefficients())
        #skip first term which is mean
        variance=np.sum(coeffs[1:]**2)
        if np.abs(ot_var-variance)>10**-6:
            raise ValueError()
        return variance
    @property
    def analytical_std(self):
        return np.sqrt(self.analytical_variance)
    
    def sample(self,num_samples,method='MC'):
        model=self._ot_res.getMetaModel()
        # need to add method here? 
        dist=self._ot_res.getDistribution()
        if method=='MC':
            experiment=ot.MonteCarloExperiment(dist,num_samples)
            self.new_sample_points=np.array(experiment.generate())
        elif method=='LHS':
            experiment=ot.LHSExperiment(dist,num_samples)
            self.new_sample_points=np.array(experiment.generate())
        elif method=='Sobol':
            sobolseq=ot.SobolSequence(dist.getDimension())
            experiment=ot.LowDiscrepancyExperiment(sobolseq,dist,num_samples)
            self.new_sample_points=np.array(experiment.generate())
        elif method=='Halton':
            haltonseq=ot.HaltonSequence(dist.getDimension())
            experiment=ot.LowDiscrepancyExperiment(haltonseq,dist,num_samples)
            self.new_sample_points=np.array(experiment.generate())

        Y=model(self.new_sample_points)
        self.new_sample_values=np.array(Y) 
        return np.array(self.new_sample_points),np.array(self.new_sample_values)
    
    def plot_cdf(self,save_dir=None):
        if self.new_sample_values is None:
            raise ValueError('Must generate samples before plotting cdf')
        vals=self.new_sample_values.flatten()
        ecdf=stats.ecdf(vals)
        quantiles=ecdf.cdf.quantiles
        probabilities=ecdf.cdf.probabilities
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(quantiles,probabilities,color='blue')   
        ax.set_title(f'CDF of surrogate output', fontsize=16,fontweight='bold')
        ax.set_xlabel('Output', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/CDF_surrogate_output.png", dpi=300, bbox_inches='tight')

        return np.hstack((quantiles,probabilities))



    def plot_pdf(self,bins=20,save_dir=None):
        if self.new_sample_values is None:
            raise ValueError('Must generate samples before plotting pdf')
        vals=self.new_sample_values.flatten()
        fig,ax=plt.subplots(figsize=(10,6))
        ax.hist(vals,bins=bins,color='blue',edgecolor='black')
        ax.set_title(f'PDF of surrogate output', fontsize=16,fontweight='bold')
        ax.set_xlabel('Output', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/PDF_surrogate_output.png", dpi=300, bbox_inches='tight')

        return ax


@register_post_workflow
class SobolSensitivityPost(BasicStatsPost):

    METHOD_TO_CLASS={
        'Saltelli':ot.SaltelliSensitivityAlgorithm,
        'Jansen':ot.JansenSensitivityAlgorithm,
        'MauntzKucherenko':ot.MauntzKucherenkoSensitivityAlgorithm,
        'Martinez':ot.MartinezSensitivityAlgorithm
        }    


    def init(self,tradestudy):
        super().init(tradestudy)
    
    def generate_Sobol_indices(self,output_name,algo='Jansen'):
        # Output values
        outs=np.array(self.tradestudy.get_variable(output_name))[:,None]

        # Overarching UQ distributor class
        UQ_Distrubutor=self.tradestudy.study.control.trade_studies[self.tradestudy.trade_name]
        aleatory_dict=UQ_Distrubutor.aleatory_var_to_dist

        #Samples run
        num_samples=UQ_Distrubutor.UQ_workflow.num_samples
        # Number of aleatory vars
        #Calculates the number of samples needed to preallocate array shape
        second_order=UQ_Distrubutor.UQ_workflow.computeSecondOrder
        num_aleatory=len(aleatory_dict)
        if not second_order or num_aleatory==2:
            num_sobol_samples=num_samples*(2+num_aleatory)
        else:
            num_sobol_samples=num_samples*(2+2*num_aleatory)

        input_matrix=np.zeros((num_sobol_samples,num_aleatory))
        name_to_col={}
        col_to_name={}
        for col, aleatory_var_name in enumerate(aleatory_dict.keys()):
            name_to_col[aleatory_var_name]=col
            col_to_name[col]=aleatory_var_name
            input_matrix[:,col]=self.tradestudy.get_variable(aleatory_var_name)
        input_matrix=ot.Sample(input_matrix)
        outs=ot.Sample(outs)
        ot_algo=self.METHOD_TO_CLASS[algo]
        otSobol=ot_algo(input_matrix,outs,num_samples)

        results={
            'first_order':{},
            'second_order':{},
            'total_order':{}
            }
        
        #First order indices, one output per input distribution
        first_order=otSobol.getFirstOrderIndices()
        for name,col in name_to_col.items():
            results['first_order'][name]=first_order[col]


        # Returns symmetric matrix (A=A^t) of all S(i,j) pairs
        if second_order:
            second_order=otSobol.getSecondOrderIndices()
            for i in range(num_aleatory):
                for j in range(i+1, num_aleatory):
                    name_i = col_to_name[i]
                    name_j = col_to_name[j]
                    results["second_order"][(name_i, name_j)] = float(second_order[i, j])

        #Total: Si+Sum(j,d)(Si,j)(j!=i)+Sum(j<=k)(Si,j,k)(i!=j<k)+...
        # one variable per input distribution
        total_order=otSobol.getTotalOrderIndices()
        for name,col in name_to_col.items():
            results['total_order'][name]=total_order[col]

        return results,otSobol

    def get_first_order_indices(self,output_name,algo='Jansen'):
        results,_=self.generate_Sobol_indices(output_name,algo)
        return results['first_order']

    def get_second_order_indices(self,output_name,algo='Jansen'):
        UQ_Distrubutor=self.tradestudy.study.control.trade_studies[self.tradestudy.trade_name]
        if not UQ_Distrubutor.UQ_workflow.computeSecondOrder:
            raise ValueError(f'This run did not support second order Sobol')
        
        results,_=self.generate_Sobol_indices(output_name,algo)

        return results['second_order']
    
    def get_total_order_indices(self,output_name,algo='Jansen'):
        results,_=self.generate_Sobol_indices(output_name,algo)
        return results['total_order']


    
    
# Define various sampling methods

class BaseSampler:
    @staticmethod
    def sample():
        pass

@register_sampler
class MonteCarloSampler(BaseSampler):

    _OT_ResourceMapName='MonteCarlo'

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        if len(aleatory_to_dist)==1:
            (name,dist),=aleatory_to_dist.items()
            var_name_to_col={name:0}
            dist=dist._OT_Dist()
        else:
            joint=[]
            var_name_to_col={}
            for i,(name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name]=i
            dist=ot.JointDistribution(joint)
        experiment=ot.MonteCarloExperiment(dist,num_samples)
        samples=np.array(experiment.generate())


        return samples,var_name_to_col
    
@register_sampler
class LatinHypercubeSampler(BaseSampler):

    _OT_ResourceMapName='LHS'

    
    @staticmethod
    def sample(aleatory_to_dist,num_samples):
        #first generate the evenly spaced values on [0,1]
        if len(aleatory_to_dist)==1:
            (name,dist),=aleatory_to_dist.items()
            var_name_to_col={name:0}
            dist=dist._OT_Dist()
        else:
            joint=[]
            var_name_to_col={}
            for i,(name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name]=i
            dist=ot.JointDistribution(joint)
        experiment=ot.LHSExperiment(dist,num_samples)
        samples=np.array(experiment.generate())


        return samples,var_name_to_col
    
@register_sampler
class SobolSampler(BaseSampler):

    #Note this use here is for SobolExperiment, where the
    #Quasi monte carlo uses the Sobol sampler
    _OT_ResourceMapName='QMC'



    @staticmethod
    def sample(aleatory_to_dist,num_samples):
        if len(aleatory_to_dist)==1:
            (name,dist),=aleatory_to_dist.items()
            var_name_to_col={name:0}
            dist=dist._OT_Dist()
        else:
            joint=[]
            var_name_to_col={}
            for i,(name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name]=i
            dist=ot.JointDistribution(joint)
        sobolseq=ot.SobolSequence(dist.getDimension())
        experiment=ot.LowDiscrepancyExperiment(sobolseq,dist,num_samples)
        samples=np.array(experiment.generate())
        
        return samples,var_name_to_col
        

@register_sampler
class HaltonSampler(BaseSampler):

    @staticmethod
    def sample(aleatory_to_dist,num_samples):
        if len(aleatory_to_dist)==1:
            (name,dist),=aleatory_to_dist.items()
            var_name_to_col={name:0}
            dist=dist._OT_Dist()
        else:
            joint=[]
            var_name_to_col={}
            for i,(name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name]=i
            dist=ot.JointDistribution(joint)
        haltonseq=ot.HaltonSequence(dist.getDimension())
        experiment=ot.LowDiscrepancyExperiment(haltonseq,dist,num_samples)
        samples=np.array(experiment.generate())
        
        return samples,var_name_to_col
    
# Define various continuous distribution classes
@register_distribution
class NormalDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _OT_Dist(self):
        return ot.Normal(self.mean,self.std)
    
    def _to_dict(self):
        return {
            'name':self.__class__.__name__,
            'mean':self.mean,
            'std':self.std
        }
    
    @classmethod
    def _from_dict(cls,data):
        return cls(data['mean'],data['std'])
        

@register_distribution
class UniformDistribution:
    def __init__(self, low, high):
        self.low = low
        self.high=high
        self.scale = self.high-self.low

    def _OT_Dist(self):
        return ot.Uniform(self.low,self.high)
    
    def _to_dict(self):
        return {
            'name':self.__class__.__name__,
            'low':self.low,
            'high':self.high}
    
    @classmethod
    def _from_dict(cls,data):
        return cls(data['low'],data['high'])


@register_distribution
class LogNormalDistribution:
    def __init__(self, mu, sigma, gamma=0.0):
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma

    def _OT_Dist(self):
        return ot.LogNormalMuSigma(self.mu, self.sigma, self.gamma).getDistribution()

    def _to_dict(self):
        return {
            'name': self.__class__.__name__,
            'mu': self.mu,
            'sigma': self.sigma,
            'gamma': self.gamma,
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(data['mu'], data['sigma'], data.get('gamma', 0.0))
    
# Pair all workflow classes with their post process classes    
pre_to_post_workflow_registry={
    'BasicStats':BasicStatsPost,
    'RegressionPCE':RegressionPCEPost,
    'SobolSensitivity':SobolSensitivityPost,
    'GaussQuadraturePCE':GaussQuadtraturePCEPost
}
if __name__ == "__main__":
    
    #Test the LHS functionality:
    uni=UniformDistribution(0.0,10.0)
    norm=NormalDistribution(50.0,25.0)

    sample_alteatory={'var1':uni,'var2':norm}
    sample_single={'var':uni}
    n=500
    x=LatinHypercubeSampler.sample(sample_alteatory,n)

    j=ot.JointDistribution([uni._OT_Dist(),norm._OT_Dist()])
    print('hello')
