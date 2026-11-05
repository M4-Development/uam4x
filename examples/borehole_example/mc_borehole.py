import numpy as np
import scipy.stats as stats
import os

def compute_borehole(rw,r,Tu,Hu,Tl,Hl,L,Kw):
        numerator=2*np.pi*Tu*(Hu-Hl)
        denominator=np.log(r/rw)*(1+(2*L*Tu)/(np.log(r/rw)*rw**2*Kw)+Tu/Tl)
        flow_rate=numerator/denominator
        return flow_rate

if __name__=="__main__":
    rng = np.random.default_rng(42)
    num_points=10**5
    rw_arr=rng.normal(loc=0.1,scale=0.0161812,size=num_points)[:,None]    
    r_arr=rng.lognormal(mean=7.71,sigma=1.0056,size=num_points)[:,None]    
    Tu_arr=rng.uniform(low=63070.,high=115600.,size=num_points)[:,None]    
    Hu_arr=rng.uniform(low=990.,high=1100.,size=num_points)[:,None]    
    Tl_arr=rng.uniform(low=63.1,high=116.,size=num_points)[:,None]    
    Hl_arr=rng.uniform(low=700.,high=820.,size=num_points)[:,None]    
    L_arr=rng.uniform(low=1120.,high=1680.,size=num_points)[:,None]    
    Kw_arr=rng.uniform(low=9985.,high=12045.,size=num_points)[:,None]    

    full_arr=np.hstack((rw_arr,r_arr,Tu_arr,Hu_arr,Tl_arr,Hl_arr,L_arr,Kw_arr))

    output=[]
    for point in full_arr:
          rw,r,Tu,Hu,Tl,Hl,L,Kw=point
          output.append(compute_borehole(rw,r,Tu,Hu,Tl,Hl,L,Kw))

    output=np.array(output)

    #basic moments
    mean=np.mean(output)
    std=np.std(output)
    variance=np.var(output)

    ##quantile prediction
    #quantiles_to_check=np.linspace(0.0,1.0,21)
    #np.quantile(output,quantiles_to_check)

    #cdf prediction
    res=stats.ecdf(output).cdf
    quantiles=res.quantiles
    probabilities=res.probabilities

    base=r'E:\NASA_UAM\uam4x_examples\borehole_example'
    full_file=os.path.join(base,'full_analysis.csv')
    
    data=np.column_stack((full_arr,output,quantiles,probabilities)) 
    np.savetxt(full_file,data,delimiter=',',header="rw,r,Tu,Hu,Tl,Hl,L,Kw,Flow_rate,quantiles,probabilities")