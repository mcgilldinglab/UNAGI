package edu.cmu.cs.sb.drem;

/**
 * Class implements a logistic regression classifier trained with an L1 penalty.
 * The L1 penalty promotes sparsity in the classifier Implementation based on
 * the following paper Krishnapuram, B., Figueiredo, M., Carin, L., and
 * Hartemink, A. Sparse Multinomial Logistic Regression: Fast Algorithms and
 * Generalization Bounds. IEEE Transactions on Pattern Analysis and Machine
 * Intelligence (PAMI), 27, June 2005. pp. 957-968.
 */
public final class DREM_FastLogistic2 {
	static final double EPSILON1 = 0.01;
	static final int MAXITERATIONS = 1000;
	static final double LAMBDA = 1;

	double[][] traindata;
	int[][] traindataindex;
	int[][] traindataTFindex;
	double[][] traindataTF;

	double[] initbeta;
	int[] y;
	double[] dyerror;
	/**
	 * In the binary case, caches the value exp(w * x) for one class and each
	 * gene. Updated as the weights are iteratively improved. Dimension:<br>
	 * [num training data points]
	 */
	double[] dipcache;
	/**
	 * In the multinomial case, caches the value exp(w * x) for each class and
	 * gene. Updated as the weights are iteratively improved. Dimensions:<br>
	 * [num classes-1][num training data points]
	 */
	double[][] dipcachemulti;
	/**
	 * Gives the likelihood of a gene going through a state, which is equivalent
	 * to the likelihood of a gene belonging to a particular class. ??? These
	 * weights are not related to the weight vector w in the paper.
	 */
	double[] dtrainweight;
	double[] dbeta;
	double dridge = LAMBDA;
	double[] dcoeff;
	int numclasses;
	int numels;
	double doldcoeff;
	int numrows;
	/**
	 * Negative definite matrix that lower bounds the Hessian of the weights???
	 * Only the diagonal is stored. ??? Dimension:<br>
	 * [(num TFs + 1)], which is d
	 */
	double[] B;
	double[] g;
	int numcols; // num features

	// miRNA new
	int[] regulatorTypes; // an arbitrary field to denote different types of
	// regulators, e.g., 0=TF and 1=miRNA for each entry
	// in RegulatorNames
	int childExpStatus;

	/**
	 * Empty constructor
	 */
	public DREM_FastLogistic2() {
	}

	/**
	 * Constructor where the coefficients are given to the classifier
	 */
	public DREM_FastLogistic2(int[][] traindataindex, double[][] traindata,
			int[][] traindataTFindex, double[][] traindataTF, int[] y,
			int numclasses, double[] dcoeff, int numbits, int[] regTypes) {
		this.traindataindex = traindataindex;
		this.traindata = traindata;
		this.traindataTF = traindataTF;
		this.traindataTFindex = traindataTFindex;
		this.numrows = traindata.length;
		this.y = y;
		this.regulatorTypes = regTypes;
		this.numclasses = numclasses;
		this.dcoeff = dcoeff;

		numcols = numbits + 1; // adding column for empty 1s
		numels = (numclasses - 1) * numcols; // one set of coefficients implicit
		g = new double[numels];
		B = new double[numcols];
	}

	/**
	 * Constructor where the training weights of the instances are given
	 */
	public DREM_FastLogistic2(int[][] traindataindex, double[][] traindata,
			int[][] traindataTFindex, double[][] traindataTF, int[] y,
			double[] dtrainweight, int numclasses, int numbits, int[] regTypes) {
		this.traindataindex = traindataindex;
		this.traindata = traindata;
		this.traindataTFindex = traindataTFindex;
		this.traindataTF = traindataTF;
		this.numrows = traindata.length;
		this.y = y;
		this.dtrainweight = dtrainweight;
		this.numclasses = numclasses;
		this.regulatorTypes = regTypes;

		dyerror = new double[dtrainweight.length];
		if (numclasses == 2) {
			dipcache = new double[dtrainweight.length];
		} else {
			dipcachemulti = new double[numclasses - 1][dtrainweight.length];
		}

		this.numcols = numbits + 1;// adding column for empty 1s
		numels = (numclasses - 1) * numcols; // one set of coefficients implicit
		g = new double[numels];
		dcoeff = new double[numels];
		B = new double[numcols];
	}

	/**
	 * Clones the classifier
	 */
	public Object clone() {
		DREM_FastLogistic2 fl = new DREM_FastLogistic2();
		fl.traindata = traindata;
		fl.traindataTF = traindataTF;
		fl.traindataindex = traindataindex;
		fl.traindataTFindex = traindataTFindex;

		fl.initbeta = initbeta;
		fl.y = y;
		fl.dtrainweight = dtrainweight;
		fl.dbeta = dbeta;
		fl.dridge = dridge;
		fl.dcoeff = new double[dcoeff.length];
		for (int nindex = 0; nindex < dcoeff.length; nindex++) {
			fl.dcoeff[nindex] = dcoeff[nindex];
		}
		fl.numclasses = numclasses;
		fl.numcols = numcols;
		fl.numels = numels;
		fl.g = g;
		fl.dyerror = dyerror;
		fl.dipcache = dipcache;
		fl.dipcachemulti = dipcachemulti;
		fl.numrows = traindata.length;
		fl.B = B;

		return fl;
	}

	/**
	 * Sets the value of the ridge parameter
	 */
	public void setRidge(double dridge) {
		this.dridge = dridge;
	}

	/**
	 * Converts the classifier into a string
	 */
	public String toString() {
		StringBuffer szbuf = new StringBuffer("Coefficients: ");
		if (dcoeff == null) {
			szbuf.append("Not initialized yet!");
		} else {
			for (int nrow = 0; nrow < dcoeff.length; nrow++) {
				szbuf.append("(" + nrow + "," + dcoeff[nrow] + ")\t");
			}
			szbuf.append("\n");
		}

		return szbuf.toString();
	}

	/**
	 * Returns a string with the classifier parameters
	 */
	public String saveClassifier(String[] tfNames) {
		StringBuffer szbuf = new StringBuffer();

		for (int nrow = 0; nrow < dcoeff.length; nrow++) {
			if (nrow % (tfNames.length + 1) == 0) {
				szbuf.append("INTERCEPT\t" + dcoeff[nrow]);
			} else {
				szbuf.append(tfNames[nrow % (tfNames.length + 1) - 1] + "\t"
						+ dcoeff[nrow]);
			}
			szbuf.append("\n");
		}

		return szbuf.toString();
	}

	/**
	 * Reinitialized the variables of the classifier based on the input
	 * parameters
	 */
	public void reinit(int[][] traindataindex, double[][] traindata,
			int[][] traindataTFindex, double[][] traindataTF, int[] y,
			double[] dtrainweight, int numclasses, int[] regTypes) {
		this.traindataindex = traindataindex;
		this.traindata = traindata;
		this.traindataTFindex = traindataTFindex;
		this.traindataTF = traindataTF;
		this.numrows = traindata.length;
		this.y = y;
		this.regulatorTypes = regTypes;

		if ((this.dtrainweight == null)
				|| (this.dtrainweight.length != dtrainweight.length)) {
			dyerror = new double[dtrainweight.length];

			if (numclasses == 2) {
				dipcache = new double[dtrainweight.length];
			} else {
				dipcachemulti = new double[numclasses - 1][dtrainweight.length];
			}
		}
		this.dtrainweight = dtrainweight;

		if (this.numclasses != numclasses) {
			this.numclasses = numclasses;
			// assuming numcols fixed

			numels = (numclasses - 1) * numcols;
			dcoeff = new double[numels];
			g = new double[numels];
		}
	}

	/**
	 * If abs(da)-ddelta less than or equal to 0 returns 0 otherwise if da is
	 * non-negative returns abs(da)-ddelta and otherwise returns
	 * -(abs(da)-ddelta)
	 */
	public double soft(double da, double ddelta) {
		double ddiff = Math.abs(da) - ddelta;
		if (ddiff <= 0) {
			return 0;
		} else {
			if (da >= 0) {
				return ddiff;
			} else {
				return (-ddiff);
			}
		}
	}

	/**
	 * A unidirectional soft thresholding prevents weight values bigger than 0,
	 * If abs(da)-ddelta greater than or equal to 0 returns 0 otherwise if da is
	 * negative returns abs(da)-ddelta
	 */
	public double unidirectionalsoft(double da, double ddelta,
			int constraintType) {
		double ddiff = Math.abs(da) - ddelta;
		/*
		 * CASE -1 0smaller1: ptr.nextptr[0].dmean < ptr.nextptr[1].dmeanchild
		 * -> coeff < 0 -> da > 0 CASE 1 1smaller0: ptr.nextptr[0].dmean >
		 * ptr.nextptr[1].dmeanchild -> coeff > 0 -> da < 0
		 */
		if (constraintType == -1) {
			if (ddiff >= 0 || da < 0) {
				return 0;
			} else {
				return -ddiff;
			}
		} else {
			if (ddiff >= 0 || da > 0) {
				return 0;
			} else {

				return ddiff;
			}

		}
	}

	/**
	 * Loads some cached values when dealing in the multiclass case
	 */
	private void fillMultiCache() {
		int numclassesm1 = numclasses - 1;
		double dlastip = Double.POSITIVE_INFINITY;
		double dval = Double.POSITIVE_INFINITY;
		for (int nrow = 0; nrow < traindata.length; nrow++) {
			double[] dx = traindata[nrow];

			for (int nindicator = 0; nindicator < numclassesm1; nindicator++) {
				int noffset = nindicator * numcols;
				double dip = dcoeff[noffset];
				int nxindex;

				// Calculate w * x
				for (int nindex = 0; nindex < traindataindex[nrow].length; nindex++) {
					nxindex = traindataindex[nrow][nindex];
					dip += dx[nindex] * dcoeff[nxindex + noffset + 1];
				}

				// Avoid calling the exp function more than what is necessary
				// by tracking the last value
				if (dlastip != dip) {
					dval = Math.exp(dip);
					dlastip = dip;
				}

				// Store exp(w * x)
				dipcachemulti[nindicator][nrow] = dval;
			}
		}
	}

	// /////////////////////////////////////////////////
	/**
	 * Computes the probability of the item in nrow is of class nindicator.
	 * Presently only used in the multi-class case.
	 */
	public double p(int nindicator, int nrow) {
		double dnum;

		double ddenom;
		if (nindicator == numclasses - 1) {
			// If the class is m-1, calculate
			// 1 / (1 + Sum_{j=0}^{m-2} exp(w_j * x)) ???
			dnum = 1;
			ddenom = 1;
		} else {
			// If the class c is not m-1, calculate
			// exp(w_c * x) / (1 + Sum_{j=0}^{m-2} exp(w_j * x)) ???
			dnum = dipcachemulti[nindicator][nrow];
			ddenom = 1 + dnum;
		}

		// can save time not recomputing current class
		for (int nj = 0; nj < numclasses - 1; nj++) {
			if (nindicator != nj) {
				ddenom += dipcachemulti[nj][nrow];
			}
		}

		double dprob = dnum / ddenom;
		return dprob;
	}

	/**
	 * Updates the G parameters in the multi-class case bnew means the
	 * coefficient changed on the previous coordinate or first time if didn't
	 * change then can keep previous errors just update current coefficient
	 * bfirsttime means it is the first pass through the coefficients
	 */
	public void computeG(int nel, boolean bnew, boolean bfirsttime) {
		// if nel == 0 and bfirsttime is true then we need to compute everything
		// otherwise we just need to do an update calc
		// See computeGbin for additional comments that are also applicable here

		int numclassesm1 = numclasses - 1;
		int nelmod = nel % numcols;

		if ((bnew) || (nelmod == 0)) {
			g[nel] = 0;

			if (nelmod == 0) {
				// need to compute everything
				double dlastip = Double.POSITIVE_INFINITY;
				double dval = Double.POSITIVE_INFINITY;
				int nclass = nel / numcols;
				for (int nrow = 0; nrow < traindata.length; nrow++) {
					double[] dx = traindata[nrow];
					for (int nindicator = 0; nindicator < numclassesm1; nindicator++) {
						int noffset = nindicator * (numcols);
						double dip = dcoeff[noffset];
						int nxindex;
						for (int nindex = 0; nindex < traindataindex[nrow].length; nindex++) {
							nxindex = traindataindex[nrow][nindex];
							dip += dx[nindex] * dcoeff[nxindex + noffset + 1];
						}

						if (dlastip != dip) {
							dval = Math.exp(dip);
							dlastip = dip;
						}
						dipcachemulti[nindicator][nrow] = dval;
					}

					double dmu = p(nclass, nrow);

					if (y[nrow] == nclass) {
						dyerror[nrow] = dtrainweight[nrow] * (1 - dmu);
					} else {
						dyerror[nrow] = -dtrainweight[nrow] * dmu;
					}
					g[nel] += dyerror[nrow];
				} // bfirsttime else
			} // nelmod == 0
			else {

				int nindicator = nel / numcols;
				// not on el 0
				if (nelmod == 1) {
					// constant always before this
					double dcoeffdiff = dcoeff[nel - 1] - doldcoeff;
					double dcoeffdiffexp = Math.exp(dcoeffdiff);
					for (int nrow = 0; nrow < traindata.length; nrow++) {
						dipcachemulti[nindicator][nrow] *= dcoeffdiffexp;

						double dmu = p(nindicator, nrow);

						// check this for multi
						if (y[nrow] == nindicator) {
							dyerror[nrow] = dtrainweight[nrow] * (1 - dmu);
						} else {
							dyerror[nrow] = -dtrainweight[nrow] * dmu;
						}
					}

					for (int ntfindex = 0; ntfindex < traindataTFindex[0].length; ntfindex++) {
						g[nel] += dyerror[traindataTFindex[0][ntfindex]]
								* traindataTF[0][ntfindex];
					}

				} // el == 1
				else {
					// not on el 0 or el 1
					// int nel2 = nel - 2;
					int nel1 = nel - 1;
					/*
					 * We need nelmod -1 to correct for the fact that the
					 * coefficient vector has the intercept at position 0. We
					 * need nelmod-2 because of the above and because we want to
					 * use the previous nel (nel-1) which leads to (nelmod1)-1.
					 */
					int nelmod1 = nelmod - 1;
					int nelmod2 = nelmod - 2;
					double dcoeffdiff = dcoeff[nel1] - doldcoeff;
					double dcoeffdiffexp = Math.exp(dcoeffdiff);

					for (int ntfindex = 0; ntfindex < traindataTFindex[nelmod2].length; ntfindex++) {
						int nrow = traindataTFindex[nelmod2][ntfindex];

						if (traindataTF[nelmod2][ntfindex] == 1) {
							dipcachemulti[nindicator][nrow] *= dcoeffdiffexp;
						} else if (traindataTF[nelmod2][ntfindex] == -1) {
							dipcachemulti[nindicator][nrow] /= dcoeffdiffexp;
						} else {
							double dprod = dcoeffdiff
									* traindataTF[nelmod2][ntfindex];
							dipcachemulti[nindicator][nrow] *= Math.exp(dprod);
						}

						double dmu = p(nindicator, nrow);

						// check this for multi
						if (y[nrow] == nindicator) {
							dyerror[nrow] = dtrainweight[nrow] * (1 - dmu);
						} else {
							dyerror[nrow] = -dtrainweight[nrow] * dmu;
						}
					}

					for (int ntfindex = 0; ntfindex < traindataTFindex[nelmod1].length; ntfindex++) {
						g[nel] += dyerror[traindataTFindex[nelmod1][ntfindex]]
								* traindataTF[nelmod1][ntfindex];
					}
				}
			}
		} else {
			// coefficient not new, use errors from last time
			g[nel] = 0;

			if (nelmod == 0) {
				for (int nrow = 0; nrow < traindata.length; nrow++) {
					g[nel] += dyerror[nrow];
				}
			} else {
				int nelmod1 = nelmod - 1;

				for (int ntfindex = 0; ntfindex < traindataTFindex[nelmod1].length; ntfindex++) {
					g[nel] += dyerror[traindataTFindex[nelmod1][ntfindex]]
							* traindataTF[nelmod1][ntfindex];
				}

			}
		}

		if (Double.isNaN(g[nel])) {
			System.out.println("computeG\t" + g[nel] + "\t" + dcoeff[nel]
					+ "\t" + dridge + "\t" + nel + "\t" + bnew + "\t"
					+ bfirsttime);
		}
	}

	/**
	 * Updates the value of the G matrix for the binary case. Based on Equation
	 * 9 in Krishnapuram et al.
	 * 
	 * @param bnew
	 *            means the coefficient changed on the previous coordinate or
	 *            first time. If didn't change then can keep previous errors
	 *            just update current coefficient
	 * @param bfirsttime
	 *            means it is the first pass through the coefficients
	 * @param nel
	 *            The index of G to be updated. Ranges from 0 to numcols-1???
	 */
	public void computeGbin(int nel, boolean bnew, boolean bfirsttime) {
		// if nel == 0 and bfirsttime is true then we need to compute everything
		// otherwise we just need to do an update calc

		int nlastcolindex = numcols - 2;
		// System.out.println("nlastcolindex = "+nlastcolindex);
		if (bnew) {
			// coefficient changed on previous iteration
			g[nel] = 0;

			if (nel == 0) {
				// the one intercept
				if (bfirsttime) {
					// need to compute everything the first time
					double dlastip = Double.POSITIVE_INFINITY;
					double dnum = Double.POSITIVE_INFINITY;
					// Iterate through all training data
					for (int nrow = 0; nrow < traindata.length; nrow++) {
						double dip = dcoeff[0]; // implicit 1 in training data
						double[] dx = traindata[nrow];
						int[] dxIndex = traindataindex[nrow];
						int nxindex;
						for (int nindex = 0; nindex < dxIndex.length; nindex++) {
							nxindex = dxIndex[nindex];
							dip += dx[nindex] * dcoeff[nxindex + 1];
						}

						if (dlastip != dip) {
							dnum = Math.exp(dip);
							dlastip = dip;
						}
						dipcache[nrow] = dnum;
						// The probability that the gene belongs to class 1
						// using the current w vector ???
						// p_j(w) in the paper ???
						double dpbinval = dnum / (1 + dnum);
						// No need to multiply by the training data because for
						// g[0] the training data is implicitly the 1s vector???
						if (y[nrow] == 0) {
							// When the jth training example belongs to class 0,
							// y'_j = 1 ???
							// trainweight * (y'_j - p_j(w))
							dyerror[nrow] = dtrainweight[nrow] * (1 - dpbinval);
						} else {
							// When the jth training example belongs to class 1,
							// y'_j = 0 ???
							// trainweight * (y'_j - p_j(w))
							dyerror[nrow] = -dtrainweight[nrow] * dpbinval;
						}
						// Sum over all training data
						g[0] += dyerror[nrow];
					}
				} else // bfirsttime is false
				{
					// already have been around at least once, updating change
					// from last coordinate
					// Need to determine what was changed when the last element
					// of the w vector
					// was updated and update the 0th element of g accordingly
					// ???
					double dcoeffdiff = dcoeff[dcoeff.length - 1] - doldcoeff;
					double dcoeffdiffexp = Math.exp(dcoeffdiff);
					int nlastrow = 0;
					for (int ntfindex = 0; ntfindex < traindataTFindex[nlastcolindex].length; ntfindex++) {
						int nrow = traindataTFindex[nlastcolindex][ntfindex];
						while (nlastrow < nrow) {
							g[0] += dyerror[nlastrow];
							nlastrow++;
						}
						nlastrow++;

						if (traindataTF[nlastcolindex][ntfindex] == 1) {
							dipcache[nrow] *= dcoeffdiffexp;
						} else if (traindataTF[nlastcolindex][ntfindex] == -1) {
							dipcache[nrow] /= dcoeffdiffexp;
						} else {
							// If the interaction value isn't 1 or -1 we can't
							// reuse
							// Math.exp(dcoeffdiff) and have to recompute the
							// exponent
							double dprod = dcoeffdiff
									* traindataTF[nlastcolindex][ntfindex];
							dipcache[nrow] *= Math.exp(dprod);
						}

						double dnum = dipcache[nrow];
						double dpbinval = dnum / (1 + dnum);
						if (y[nrow] == 0) {
							dyerror[nrow] = dtrainweight[nrow] * (1 - dpbinval);
						} else {
							dyerror[nrow] = -dtrainweight[nrow] * dpbinval;
						}
						g[0] += dyerror[nrow]; // update score for all values
					}

					while (nlastrow < traindata.length) {
						g[0] += dyerror[nlastrow];
						nlastrow++;
					}
				} // end bfirsttime is false
			} else {
				// Separate cases for nel = 1 and nel > 1 the last element of
				// the w vector to be updated (nel-1) is used to determine
				// how to update the current element, but nel = 0 must be
				// handled differently because its training data (the 1s
				// vector) is not explicilty stored
				// not on el 0
				if (nel == 1) {
					// constant always before this
					// i.e. the last coefficient to be updated was that for the
					// intercept???
					double dcoeffdiff = dcoeff[0] - doldcoeff;
					double dcoeffdiffexp = Math.exp(dcoeffdiff);
					for (int nrow = 0; nrow < traindata.length; nrow++) {
						// The interaction value (training data value) is
						// implicitly 1 ???
						dipcache[nrow] *= dcoeffdiffexp;

						double dnum = dipcache[nrow];
						double dpbinval = dnum / (1 + dnum);

						if (y[nrow] == 0) {
							dyerror[nrow] = dtrainweight[nrow] * (1 - dpbinval);
						} else {
							dyerror[nrow] = -dtrainweight[nrow] * dpbinval;
						}
					}

					for (int ntfindex = 0; ntfindex < traindataTFindex[0].length; ntfindex++) {
						g[nel] += dyerror[traindataTFindex[0][ntfindex]]
								* traindataTF[0][ntfindex];
					}
				} else {
					// not on el 0 or el 1
					int nel2 = nel - 2;
					int nel1 = nel - 1;
					double dcoeffdiff = dcoeff[nel1] - doldcoeff;
					double dcoeffdiffexp = Math.exp(dcoeffdiff);
					for (int ntfindex = 0; ntfindex < traindataTFindex[nel2].length; ntfindex++) {
						int nrow = traindataTFindex[nel2][ntfindex];
						if (traindataTF[nel2][ntfindex] == 1) {
							dipcache[nrow] *= dcoeffdiffexp;
						} else if (traindataTF[nel2][ntfindex] == -1) {
							dipcache[nrow] /= dcoeffdiffexp;
						} else {
							double dprod = dcoeffdiff
									* traindataTF[nel2][ntfindex];
							dipcache[nrow] *= Math.exp(dprod);
						}

						double dnum = dipcache[nrow];
						double dpbinval = dnum / (1 + dnum);

						if (y[nrow] == 0) {
							dyerror[nrow] = dtrainweight[nrow] * (1 - dpbinval);
						} else {
							dyerror[nrow] = -dtrainweight[nrow] * dpbinval;
						}
					}

					for (int ntfindex = 0; ntfindex < traindataTFindex[nel1].length; ntfindex++) {
						g[nel] += dyerror[traindataTFindex[nel1][ntfindex]]
								* traindataTF[nel1][ntfindex];
					}
				}
			}
		} else // bnew is false
		{
			// coefficient not new, use errors from last time
			g[nel] = 0;

			if (nel == 0) {
				// The data vector is assumed to be all 1s for the intercept
				for (int nrow = 0; nrow < traindata.length; nrow++) {
					g[0] += dyerror[nrow];
				}
			} else {
				// The training data indices are offset by 1 because g[0]
				// is reserved for the intercept ???
				int nel1 = nel - 1;

				for (int ntfindex = 0; ntfindex < traindataTFindex[nel1].length; ntfindex++) {
					g[nel] += dyerror[traindataTFindex[nel1][ntfindex]]
							* traindataTF[nel1][ntfindex];
				}
			}
		}
                
                /*
		if (Double.isNaN(g[nel])) {
			System.out.println("computeG\t" + g[nel] + "\t" + dcoeff[nel]
					+ "\t" + (dcoeff[nel - 1] - doldcoeff) + "\t" + dridge
					+ "\t" + nel + "\t" + bnew + "\t" + bfirsttime);
		}
                **/
	}

	/**
	 * For the multi-class case updates the value of dcoeff[nel]
	 */
	public boolean nextw(int nel, boolean bnew, boolean bfirst) {
		// See nextwbin for additional comments that are also applicable here

		computeG(nel, bnew, bfirst);
		doldcoeff = dcoeff[nel];
		int nbindex = nel % numcols;

		if (B[nbindex] == 0) {
			dcoeff[nel] = 0;
		} else {
			if (nel % numcols == 0) {
				dcoeff[nel] = dcoeff[nel] - g[nel] / B[nbindex];
			} else {
				if (regulatorTypes[nbindex - 1] == RegulatorBindingData.TF) { // if
																				// TF
					dcoeff[nel] = soft(dcoeff[nel] - g[nel] / B[nbindex],
							-dridge / B[nbindex]);
				} else { // if miRNA

					dcoeff[nel] = unidirectionalsoft(dcoeff[nel] - g[nel]
							/ B[nbindex], -dridge / B[nbindex], 1);
					// check if old coefficient is "better" HERE we would ned to
					// evaluate the objective function
					// if(doldcoeff < 0 && dcoeff[nel] == 0){
					// but then we could never come back to 0...
					// dcoeff[nel]=doldcoeff;

					// }
				}

			}
		}
		if (Math.abs(dcoeff[nel]) > 10) {
//			System.out.println(" dcoeff of element:" + nel + " was "
//					+ dcoeff[nel]);
			// dcoeff[nel] = 10*Math.signum(dcoeff[nel]);
			dcoeff[nel] = 0;

		}
		if (Double.isNaN(dcoeff[nel])) {
			System.out.println("nextw\t" + dcoeff[nel] + "\t" + g[nel] + "\t"
					+ B[nbindex] + "\t" + dridge);
		}
		return (doldcoeff != dcoeff[nel]);
	}

	/**
	 * For the binary case updates the value of dcoeff[nel]. Based on Equation
	 * 15 in Krishnapuram et al.
	 */
	public boolean nextwbin(int nel, boolean bnew, boolean bfirst) {
		// returns true if change
		computeGbin(nel, bnew, bfirst);

		doldcoeff = dcoeff[nel];
		// If B[nel] is 0 it means that when B was computed, the
		// corresponding TF did not bind any genes. Therefore,
		// it's coefficient should be 0 ???
		if (B[nel] == 0) {
			dcoeff[nel] = 0;
		} else {
			// temp ridge on
			if (nel == 0) {
				// The intercept coefficient is not subject to the Laplacian
				// prior ???
				// Instead update using Equation 7 from Krishnapuram et al.???
				dcoeff[nel] = dcoeff[nel] - g[nel] / B[nel];
			} else {
				if (regulatorTypes[nel - 1] == RegulatorBindingData.TF) {
					dcoeff[nel] = soft(dcoeff[nel] - g[nel] / B[nel], -dridge
							/ B[nel]);
				} else { // if miRNA
					/*
					 * Now use a revised version that decides based on the
					 * childExpressionStatus how and if the classifier is
					 * constraining in a certain direction 
					 * CASE -1 0smaller1: ptr.nextptr[0].dmean < ptr.nextptr[1].dmeanchild -> coeff
					 * < 0 
					 * CASE +1 1smaller0: ptr.nextptr[0].dmean >
					 * ptr.nextptr[1].dmeanchild -> coeff > 0 
					 * CASE +2 bothUp: if miRNA downregulated -> coeff [-inf,+inf] 
					 * CASE -2 bothDown: if miRNA upregulated -> coeff [-inf,+inf]
					 */
					switch (childExpStatus) {
					case -1:
						dcoeff[nel] = unidirectionalsoft(dcoeff[nel] - g[nel]
								/ B[nel], -dridge / B[nel], childExpStatus);
						break;
					case 1:
						dcoeff[nel] = unidirectionalsoft(dcoeff[nel] - g[nel]
								/ B[nel], -dridge / B[nel], childExpStatus);
						break;
					case 2:
						if (traindataTF[nel - 1][0] < 0) {
							dcoeff[nel] = soft(dcoeff[nel] - g[nel] / B[nel],
									-dridge / B[nel]);
						} else {
							dcoeff[nel] = 0;
						}
						break;
					case -2:
						if (traindataTF[nel - 1][0] > 0) {
							dcoeff[nel] = soft(dcoeff[nel] - g[nel] / B[nel],
									-dridge / B[nel]);
						} else {
							dcoeff[nel] = 0;
						}
						break;
					default:
						System.out
								.println("No valid case for childExpStatus in nextwbin");
						System.exit(1);
					}
					// dcoeff[nel] = soft(dcoeff[nel] - g[nel]/B[nel],
					// -dridge/B[nel]);
					// check if old coefficient is "better"
					// if(doldcoeff < 0 && dcoeff[nel] == 0){
					// but then we could never come back to 0...
					// dcoeff[nel]=doldcoeff;

					// }
				}
			}
		}
	
		if (Math.abs(dcoeff[nel]) > 10) {
			//System.out.println(" dcoeff of element:" + nel + " was "
			//		+ dcoeff[nel]);
			// dcoeff[nel] = 10*Math.signum(dcoeff[nel]);
			dcoeff[nel] = 0;

		}
		
		return (doldcoeff != dcoeff[nel]);
	}

	// ///////////////////////////////////////////////////////////////////
	/**
	 * Computes the diagonal of the B array. Based on Equation 8 from
	 * Krishnapuram et al.
	 */
	private void computeB() {
		for (int ncol = 0; ncol < numcols; ncol++) {
			B[ncol] = 0;
		}

		for (int nrow = 0; nrow < traindataindex.length; nrow++) {
			B[0] += dtrainweight[nrow];

			// trainweight * x_j * (x_j)^T
			for (int nrowindex = 0; nrowindex < traindataindex[nrow].length; nrowindex++) {
				B[traindataindex[nrow][nrowindex] + 1] += traindata[nrow][nrowindex]
						* traindata[nrow][nrowindex] * dtrainweight[nrow];
			}
		}

		// Every element on the diagonal of the matrix formed by:
		// -1/2 * [I - 1*(1)^T / m]
		// has value dmI. Based on the first part of Equation 8 in Krishnapuram
		// et al.
		double dmI = -0.5 + 0.5 / numclasses;
		for (int ncol = 0; ncol < B.length; ncol++) {
			B[ncol] *= dmI;
		}
	}

	// ///////////////////////////////////////////////////////////////
	// Unlike the other version of distributionForInstance, this
	// version takes an int array as a parameter.
	// It's fine with ints now with the way that it is used since the user
	// can only specify binary or ternary values for the TF binding (as opposed
	// to arbitrary values) ???
	/**
	 * Returns a probability distribution the classifier gives to each class for
	 * the provided instance
	 */
	public double[] distributionForInstance(int[] theInstance) {
		double[] dist = new double[numclasses];
		if (numclasses == 2) {
			// binary for now need to generalize
			double dsum = dcoeff[0];
			for (int j = 0; j < theInstance.length; j++) {
				dsum += dcoeff[j + 1] * theInstance[j];
			}
			double dmu = 1 / (1 + Math.exp(-dsum));
			dist[0] = dmu;
			dist[1] = 1 - dmu;
		} else {
			double ddenom = 1;

			int ncoeffindex = 0;
			for (int nclass = 0; nclass < numclasses - 1; nclass++) {
				double dip = dcoeff[ncoeffindex];
				ncoeffindex++;
				for (int nindex = 0; nindex < theInstance.length; nindex++) {
					dip += theInstance[nindex] * dcoeff[ncoeffindex];
					ncoeffindex++;
				}
				double dtempval = Math.exp(dip);
				dist[nclass] = dtempval;
				ddenom += dtempval;
			}

			for (int nindex = 0; nindex < dist.length - 1; nindex++) {
				dist[nindex] /= ddenom;
			}
			dist[dist.length - 1] = 1.0 / ddenom;
		}

		return dist;
	}

	// ////////////////////////////////////////////////////////////////////////////
	/**
	 * Returns a probability distribution the classifier gives to each class for
	 * the provided instance
	 */
	public double[] distributionForInstance(int[] theInstanceIndex,
			double[] theInstance) {
		double[] dist = new double[numclasses];
		if (numclasses == 2) {

			double dsum = dcoeff[0];
			int nxindex;
			for (int j = 0; j < theInstanceIndex.length; j++) {
				nxindex = theInstanceIndex[j];
				dsum += dcoeff[nxindex + 1] * theInstance[j];
			}

			double dmu = 1 / (1 + Math.exp(-dsum));
			dist[0] = dmu;
			dist[1] = 1 - dmu;
		} else {
			double ddenom = 1;
			for (int nclass = 0; nclass < numclasses - 1; nclass++) {
				int noffset = nclass * numcols;
				double dip = dcoeff[noffset];
				int nxindex;
				for (int nindex = 0; nindex < theInstanceIndex.length; nindex++) {
					nxindex = theInstanceIndex[nindex];
					dip += theInstance[nindex] * dcoeff[nxindex + noffset + 1];
				}

				double dtempval = Math.exp(dip);
				dist[nclass] = dtempval;
				ddenom += dtempval;
			}

			for (int nindex = 0; nindex < dist.length - 1; nindex++) {
				dist[nindex] /= ddenom;
			}
			dist[dist.length - 1] = 1.0 / ddenom;
		}

		return dist;
	}

	/**
	 * Calls either trainbin or trainmulti to train the parameters of the
	 * classifier
	 */
	public void train() {
		if (numclasses == 2) {
			trainbin();
		} else {
			trainmulti();
		}
	}

	// /////////////////////////////////////////////////////////
	/**
	 * Trains the parameters in the multi-class case
	 */
	public void trainmulti() {

		computeB();
		fillMultiCache();
		double dolddev = 0;

		for (int nrow = 0; nrow < numrows; nrow++) {
			double dmu = p(y[nrow], nrow);
			dolddev += dtrainweight[nrow] * Math.log(dmu);
		}

		int niteration = 0;
		double dratio;
		boolean bfirst = true;

		do {
			boolean bnew = true;
			for (int nel = 0; nel < numels; nel++) {
				bnew = nextw(nel, bnew, bfirst);
			}

			double dev = 0;
			for (int nrow = 0; nrow < numrows; nrow++) {
				double dmu = p(y[nrow], nrow);
				dev += dtrainweight[nrow] * Math.log(dmu);
			}

			dratio = Math.abs(dolddev - dev) / (double) Math.abs(dev);
			dolddev = dev;
			niteration++;
		} while ((dratio > EPSILON1) && (niteration <= MAXITERATIONS));
	}

	// ///////////////////////////////////////////////////////
	/**
	 * Trains the parameters in the binary case
	 */
	public void trainbin() {
		// The B matrix only needs to be calculated once since it depends only
		// on
		// m and x
		computeB();
		double dolddev = 0;

		for (int nrow = 0; nrow < numrows; nrow++) {
			double dip = dcoeff[0];

			for (int ngeneindex = 0; ngeneindex < traindataindex[nrow].length; ngeneindex++) {
				int ntf = traindataindex[nrow][ngeneindex];
				dip += traindata[nrow][ngeneindex] * dcoeff[ntf + 1];
			}

			double dmu = 1 / (1 + Math.exp(-dip));
			if (y[nrow] == 1) {
				dolddev += dtrainweight[nrow] * Math.log(1 - dmu);
			} else {
				dolddev += dtrainweight[nrow] * Math.log(dmu);
			}
		}

		int niteration = 0;
		double dratio;

		boolean bfirst = true;

		do {
			boolean bnew = true;
			for (int nel = 0; nel < numcols; nel++) {
				bnew = nextwbin(nel, bnew, bfirst);
			}
			bfirst = false;

			double dev = 0;
			for (int nrow = 0; nrow < numrows; nrow++) {
				double dip = dcoeff[0];
				double[] dx = traindata[nrow];
				int[] dxIndex = traindataindex[nrow];
				int nxindex;
				for (int ni = 0; ni < dxIndex.length; ni++) {
					nxindex = dxIndex[ni];
					dip += dx[ni] * dcoeff[nxindex + 1];
				}

				double dmu = 1 / (1 + Math.exp(-dip));

				if (y[nrow] == 1) {
					dev += dtrainweight[nrow] * Math.log(1 - dmu);
				} else {
					dev += dtrainweight[nrow] * Math.log(dmu);
				}
			}

			dratio = Math.abs((dolddev - dev) / dev);

			dolddev = dev;
			niteration++;
			if(niteration==MAXITERATIONS){System.out.println("maxiterations hit.");}
		} while ((dratio > EPSILON1) && (niteration <= MAXITERATIONS));
	}
}
