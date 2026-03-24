package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import java.io.*;

/**
 * Class extends DataSetCore which contains the underlying data and parameters
 * of the methods with an instance of DREM_GoAnnotations
 */
public class DREM_DataSet extends DataSetCore {

	DREM_GoAnnotations tga;

	/**
	 * Constructor that does a simple copy the contents of theDataSetCore and
	 * tga
	 */
	public DREM_DataSet(DataSetCore theDataSetCore, DREM_GoAnnotations tga) {
		super(theDataSetCore);
		this.tga = tga;
	}

	/**
	 * Constructor that takes input parameters and calls dataSetReader to read
	 * in the content of szInputFile
	 */
	public DREM_DataSet(String szInputFile, int nmaxmissing,
			double dthresholdvalue, double dmincorrelation, boolean btakelog,
			boolean bspotincluded, boolean brepeatset, boolean badd0,
			boolean bmaxminval, boolean bfullrepeat)
			throws IOException, FileNotFoundException, IllegalArgumentException {
		this.szInputFile = szInputFile;
		this.bfullrepeat = bfullrepeat;
		this.nmaxmissing = nmaxmissing;
		this.dthresholdvalue = dthresholdvalue;
		this.dmincorrelation = dmincorrelation;
		this.bmaxminval = bmaxminval;
		this.btakelog = btakelog;
		this.bspotincluded = bspotincluded;
		this.badd0 = badd0;

		dataSetReader(szInputFile, nmaxmissing, dthresholdvalue,
				dmincorrelation, btakelog, bspotincluded, brepeatset, badd0);
	}
}