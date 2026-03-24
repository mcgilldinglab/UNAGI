package edu.cmu.cs.sb.core;
import java.util.*;

/**
 *The class implements static methods to compute values related to
 *the binomial, hypergeometric, and normal distribution
 * @author Jason Ernst
 */
public class StatUtil
{


    /**
     *this hashtable caches previously computed binomial coefficient values so they do not need to be recomputed
     */
    public static Hashtable htbinom =new Hashtable();

    /**
     *Sqrt of 2 times PI
     */
    public static double TWOPISQRT = Math.sqrt(2*Math.PI);


    /**
     *Returns the log of the binomial coefficient N choose ni
     */
    public static double logbinomcoeff(int ni, int N)
    {
	String sz = ni+";"+N;
	Double dobj = (Double) htbinom.get(sz);
        double dsum;

	if (dobj != null)
       	{
	   dsum = ((Double) dobj).doubleValue();
       	}
        else
	{
           dsum = 0;
           int dmax = Math.max(ni,N-ni);
           int dmin = Math.min(ni,N-ni);

	   //the log of the part of the numerator not cancelled by 
	   //the larger factorial in the denominator
           for (int nj = dmax+1; nj <=N; nj++)
           {
              dsum += Math.log(nj);
           }

	   //subtract off the log of the denominator of the smaller term
           for (int nj = 2; nj <=dmin; nj++)
           {
              dsum -= Math.log(nj);
           }
	   //store it
           htbinom.put(sz,new Double(dsum));
       }   

       return dsum;
    }


    /**
     * Returns the probability of seeing x of type A, when there nA objects of type A
     * nB objects of type B, and nm objects total drawn
     */
    public static double hypergeometric(int nx, int nA, int nB, int nm)
    {
       if (nx < 0)
       {
          return 0;
       }
        
       double dprob = 0;
      
       int nminx = Math.max(nm-nB, 0); //how many must be of type A
       int nmaxx = Math.min(nx, nA); //the most that can be of type A
    
       if (nminx > nmaxx)
       {
          return 0;
       }

       double dsum = -logbinomcoeff(nm, nA + nB)+ logbinomcoeff(nx, nA)
	             + logbinomcoeff(nm-nx, nB);


       dprob = Math.pow(Math.E,dsum);

       if (dprob <=0)
       {
          dprob = 0;
       }
       else if (dprob >= 1)
       {
          dprob = 1;
       }
	
       return dprob;
    }

    /**
     * Returns the probability of seeing more than x  objects of type A, when there are nA objects of type A
     * nB objects of type B, and nm objects total drawn
     * This can be used to compute a more accurate p-values than a 1-cumulative probability calculation
     */
    public static double hypergeometrictail(int nx, int nA, int nB, int nm)
    {
       if (nx < 0)
       {
	  return 1;
       }
        
       double dprob = 0;
      
   
       int nminx = Math.max(nx+1,0); //the first element to the right of x or 0
       int nmaxx = Math.min(nA,nm);  //the max of type A there can be 

       if (nminx > nmaxx)
       {
	   //if nx approaches infinity tail should be 0
	   return 0;
       }

       double dsum = -logbinomcoeff(nm, nA + nB)+ logbinomcoeff(nminx, nA)
	             + logbinomcoeff(nm-nminx, nB);

       double dlogprob=dsum;
       for (int ni = nminx+1; ni <= nmaxx; ni++)
       {
	   //computing the increase in probability mass
	   //numerator has nA!/(ni!(nA-ni)!) * nB!/((nm-ni)!(nB-nm+ni)!)
	   //denominator has (nA+nB)!/(nm!(nA+nB-nm)!)

	   //numerator has nA!/((ni-1)!(nA-ni+1)!) * nB!/((nm-ni+1)!(nB-nm+ni-1)!)
	   //denominator has (nA+nB)!/(nm!(nA+nB-nm)!)

	   //cancelling gives
	   //1/(ni!(nA-ni)!) * 1/((nm-ni)!(nB-nm+ni)!) over
	   //1/((ni-1)!(nA-ni+1)!) * 1/((nm-ni+1)!(nB-nm+ni-1)!)
           dsum  += Math.log(nA-ni+1)-Math.log(nB-nm+ni)
	         + Math.log(nm-ni+1)-Math.log(ni);


	  //log(a+b+c+d+e)
	  //log(e) + log(a+b+c+d+e) - log(e)
	  //log(e) + log((a+b+c+d+e)/e)
	  //log(e) + log(1+(a+b+c+d)/e)
	  //log(e) + log(1+Math.exp(log(a+b+c+d)-log(e)))
   
	  if (dsum >= dlogprob)
	  {   
	     dlogprob = dsum + Math.log(1+Math.pow(Math.E,dlogprob-dsum));     
	  }
	  else
	  {
	     dlogprob = dlogprob + Math.log(1+Math.pow(Math.E,dsum-dlogprob)); 
	  }
       }

       dprob = Math.pow(Math.E,dlogprob);


       if (dprob <= 0)
       {
	   return 0;
       }
       else if (dprob >= 1)
       {
	   return 1;
       }
       else
       {
	   return dprob;
       }
    }


    /**
     * Returns the probability of seeing x or fewer objects of type A, when there are nA objects of type A
     * nB objects of type B, and nm objects total drawn
     */
    public static double hypergeometriccumulativex(int nx, int nA, int nB, int nm)
    {
       if (nx < 0)
       {
          return 0;
       }
        
       double dprob = 0;
      
       int nminx = Math.max(nm-nB, 0); //how many must be of type A
       int nmaxx = Math.min(nx, nA); //the most that can be of type A
   
       if (nminx > nmaxx)
       {	   
	   //if nx less < 0 cumulative should be 0
          return 0;
       }

       double dsum = -logbinomcoeff(nm, nA + nB)+ logbinomcoeff(nminx, nA)
	             + logbinomcoeff(nm-nminx, nB);

       dprob = Math.pow(Math.E,dsum);
       for (int ni = nminx+1; ni <= nmaxx; ni++)
       {
	   //computing the increase in probability mass
	   //numerator has nA!/(ni!(nA-ni)!) * nB!/((nm-ni)!(nB-nm+ni)!)
	   //denominator has (nA+nB)!/(nm!(nA+nB-nm)!)
	   //numerator has nA!/((ni-1)!(nA-ni+1)!) * nB!/((nm-ni+1)!(nB-nm+ni-1)!)

	   //denominator has (nA+nB)!/(nm!(nA+nB-nm)!)
	   //cancelling gives
	   //1/(ni!(nA-ni)!) * 1/((nm-ni)!(nB-nm+ni)!) over
	   //1/((ni-1)!(nA-ni+1)!) * 1/((nm-ni+1)!(nB-nm+ni-1)!)

          dsum  += Math.log(nA-ni+1)-Math.log(nB-nm+ni)
	         + Math.log(nm-ni+1)-Math.log(ni);
          
          dprob += Math.pow(Math.E,dsum);
       }

       if (dprob <= 0)
       {
          dprob = 0;
       }
       else if (dprob >= 1)
       {
          dprob = 1;
       }
	
       return dprob;
    }

    /**
     *Computes the probability of seeing x or fewer successes in dN trials 
     *where the probability of a success is dp.
     */
    public static double binomialcumulative(double x, double dN, double dp)
    {
	//returns the amount of probability less than or equal to x
            
	if (x > dN)
	{
	    return 1;
	}
	else if ((x < 0)||(dp<=0)||(dp>=1))
	{
	    return 0;
	}

        int N = (int) Math.ceil(dN);
        double dterm = logbinomcoeff(0, N);
        double dpv1 = Math.log(dp);
        double dpv2 = Math.log(1-dp);

        dterm += N*dpv2;
        double dprob = Math.pow(Math.E,dterm);
	double dpdiff =  dpv1-dpv2;
        for (int ni = 1; ni <= x; ni++)
	{
	    //N!/(ni!(N-ni)!)
	    //N!/((ni-1)!(N-ni+1)!)
	   dterm += Math.log(N-ni+1) - Math.log(ni);
	   dterm += dpdiff;
           dprob += Math.pow(Math.E,dterm);
	}
        if (dprob <= 0)
	{
           dprob =0;
	}
        else if (dprob >= 1)
	{
           dprob =1;
	}

        return dprob;
    }


    /**
     *Computes the probability of seeing more than x successes in N trials 
     *where the probability of a success is dp.
     */
    public static double binomialtail(int x,int N, double dp)
    {        
	if (x > N)
	{
	    return 0;
	}  
	else if ((x < 0)||(dp<=0)||(dp>=1))
	{
	    return 1;
	}

	x++;
        double dterm = logbinomcoeff(x, N);
        double dpv1 = Math.log(dp);
        double dpv2 = Math.log(1-dp);

        dterm += x*dpv1+(N-x)*dpv2;

	double dlogprob = dterm;
	double dpdiff =  dpv1-dpv2;
	double dprob;
        for (int ni = x+1; ni <= N; ni++)
	{
	    //N!/(ni!(N-ni)!)
	    //N!/((ni-1)!(N-ni+1)!)
	   dterm += Math.log(N-ni+1) - Math.log(ni) + dpdiff;

	  if (dterm >= dlogprob)
	  {   
	     dlogprob = dterm + Math.log(1+Math.pow(Math.E,dlogprob-dterm));     
	  }
	  else
	  {
	     dlogprob = dlogprob + Math.log(1+Math.pow(Math.E,dterm-dlogprob)); 
	  }
	}
        dprob = Math.pow(Math.E,dlogprob);

       if (dprob <= 0)
       {
	   return 0;
       }
       else if (dprob >= 1)
       {
	   return 1;
       }
       else
       {
	   return dprob;
       }
    }


    /**
     *computes the value of f(x) where f is a density for a normal distribution with
     *mean dmu and standard deviation dsigma. 
     */
    public static double normaldensity(double x, double dmu, double dsigma)
    {
	double dens;
	double dxmudiff = (x-dmu);
	dens = Math.exp(-dxmudiff*dxmudiff/(2*dsigma*dsigma))/(dsigma*TWOPISQRT);

	return dens;
    }

}