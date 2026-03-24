package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;

/**
 * Implements a naive bayes classifier This class is used for predicting whether
 * a gene with a specified set of regulators will be filtered or not.
 */
public class DREM_NaiveBayes {
	int numcols;
	int numclasses;
	double[] pClass;
	int noffset;

	// p(C|F_1,...,F_n)
	// P(C,F_1,...,F_n)/P(F_1,...,F_n)
	// P(C)P(F_1|C)...P(F_n|C)/(P(C=1)P(F_1,...,F_n|C=1)+P(C=0)P(F_1,...,F_n|C=0))
	/**
	 * first class, second feature, third feature val. Stores P(F_i=f|C=c) for
	 * all values of c, i, and f ???
	 */
	double[][][] pNB;
	// nBaseCount is used in many classes outside of DREM_NaiveBayes
	/** first feaure, second feature val */
	int[][] nBaseCount;

	/**
	 * Class constructor - builds the classifier
	 */
	public DREM_NaiveBayes(int[][] traindata, int[][] traindataIndex,
			int numcols, int[] y, int[] featurevals, int numclasses) {
		this.numclasses = numclasses;
		this.numcols = numcols;
		int numrows = traindata.length;
		int numfeaturevals = featurevals.length;
		pNB = new double[numclasses][numcols][numfeaturevals];
		nBaseCount = new int[numcols][numfeaturevals];
		pClass = new double[numclasses];
		int nminval = featurevals[0];
		for (int nindex = 1; nindex < featurevals.length; nindex++) {
			if (featurevals[nindex] < nminval) {
				nminval = featurevals[nindex];
			}
		}
		noffset = -nminval;
		for (int nrow = 0; nrow < numrows; nrow++) {
			int ntfindex = 0;
			for (int ncol = 0; ncol < numcols; ncol++) {
				while ((ntfindex < traindataIndex[nrow].length)
						&& (ncol > traindataIndex[nrow][ntfindex])) {
					ntfindex++;
				}

				int nfeatureindex;
				if ((ntfindex < traindataIndex[nrow].length)
						&& (ncol == traindataIndex[nrow][ntfindex])) {
					nfeatureindex = traindata[nrow][ntfindex] + noffset;
					// System.out.println("A\t"+nfeatureindex+"\t"+numfeaturevals+"\t"+noffset+"\t"+ntfindex);
				} else {

					nfeatureindex = noffset;
					// System.out.println("B\t"+nfeatureindex+"\t"+numfeaturevals+"\t"+noffset+"\t"+ntfindex);
				}

				if ((nfeatureindex >= 0) && (nfeatureindex < numfeaturevals)) {
					// update 7/5/09
					// possible to have in data values for features that have
					// been filtered
					// we are not interested in making predictions for these
					// values and will filter since have prob. 0
					pNB[y[nrow]][ncol][nfeatureindex]++;
					nBaseCount[ncol][nfeatureindex]++;
				}
			}
			pClass[y[nrow]]++;
		}

		for (int nclass = 0; nclass < numclasses; nclass++) {
			for (int ncol = 0; ncol < numcols; ncol++) {
				for (int nfeatureval = 0; nfeatureval < numfeaturevals; nfeatureval++) {
					pNB[nclass][ncol][nfeatureval] /= pClass[nclass];
				}
			}
			pClass[nclass] /= numrows;
		}
	}

	/**
	 * Returns the probability of each class for the feature values in
	 * theInstance under the naive bayes models.
	 */
	public double[] distributionForInstance(int[] theInstance) {
		double[] dist = new double[numclasses];
		double dsum = 0;
		for (int nclass = 0; nclass < numclasses; nclass++) {
			dist[nclass] = pClass[nclass];
			for (int nfeature = 0; nfeature < numcols; nfeature++) {
				dist[nclass] *= pNB[nclass][nfeature][theInstance[nfeature]
						+ noffset];
			}
			dsum += dist[nclass];
		}

		for (int nclass = 0; nclass < numclasses; nclass++) {
			if (dist[nclass] != 0) {
				dist[nclass] /= dsum;
			}
		}

		return dist;
	}
}
