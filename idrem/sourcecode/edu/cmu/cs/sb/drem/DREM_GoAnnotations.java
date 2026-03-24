package edu.cmu.cs.sb.drem;

import java.util.*;
import java.io.*;
import edu.cmu.cs.sb.core.*;

/**
 * Class encapsules the Gene Ontology annotations and gene set enrichment
 * functions used by DREM
 */
public class DREM_GoAnnotations extends GoAnnotations {

	public RecIDdrem[] theRecIDdrem;

	/**
	 * Class constructor - calls the super class constructor
	 */
	DREM_GoAnnotations(String szorganismsourceval, String szxrefsourceval,
			String szxrefval, String szGoFile, String szGoCategoryFile,
			String[] baseGenes1, String[] baseProbes1, int nsamplespval,
			int nmingo, int nmingolevel, String szextraval,
			String szcategoryIDval, boolean bspotincluded,
			String szevidenceval, String sztaxonval, boolean bpontoval,
			boolean bcontoval, boolean bfontoval, boolean brandomgoval)
			throws FileNotFoundException, IOException, IllegalArgumentException {

		super(szorganismsourceval, szxrefsourceval, szxrefval, szGoFile,
				szGoCategoryFile, baseGenes1, baseProbes1, nsamplespval,
				nmingo, nmingolevel, szextraval, szcategoryIDval,
				bspotincluded, szevidenceval, sztaxonval, bpontoval, bcontoval,
				bfontoval, brandomgoval, null);

	}

	/**
     *
     */
	public void buildRecDREM(DREM_Timeiohmm.Treenode treeptr, String[] genenames) {
		HashMap htGOtoMinSplit = new HashMap();
		HashMap htGOtoMinOverall = new HashMap();

		findmins(htGOtoMinSplit, htGOtoMinOverall, treeptr, genenames, null);
		Set entrySet = htGOtoMinOverall.entrySet();
		Iterator itrSet = entrySet.iterator();

		theRecIDdrem = new RecIDdrem[htGOtoMinOverall.size()];

		int nindex = 0;
		while (itrSet.hasNext()) {
			Map.Entry themap = (Map.Entry) itrSet.next();
			String szID = (String) themap.getKey();
			String szName = ((Rec) htGO.get(szID)).sztermName;
			double dminoverallpval = ((Double) themap.getValue()).doubleValue();
			double dminsplitpval;
			Double dobj = (Double) htGOtoMinSplit.get(szID);
			if (dobj != null) {
				dminsplitpval = dobj.doubleValue();
			} else {
				dminsplitpval = 1;
			}
			theRecIDdrem[nindex] = new RecIDdrem(szID, szName, dminoverallpval,
					dminsplitpval);
			nindex++;

		}

		Arrays.sort(theRecIDdrem, new RecIDdremCompare());
	}

	/**
	 * Determines the minimum overall and split p-values for each
	 */
	public void findmins(HashMap htGOtoMinSplit, HashMap htGOtoMinOverall,
			DREM_Timeiohmm.Treenode treeptr, String[] genenames,
			HashMap htParentGO) {
		if (treeptr != null) {
			HashMap htGOtoCount = new HashMap();
			for (int ngene = 0; ngene < treeptr.bInNode.length; ngene++) {
				if (treeptr.bInNode[ngene]) {
					HashSet hsGO = labelsForID(genenames[ngene]);
					Iterator theiterator = hsGO.iterator();

					while (theiterator.hasNext()) {
						String szid = (String) theiterator.next();
						Integer nobjcount = (Integer) htGOtoCount.get(szid);

						if (nobjcount == null) {
							htGOtoCount.put(szid, new Integer(1));
						} else {
							htGOtoCount.put(szid, new Integer(nobjcount
									.intValue() + 1));
						}
					}
				}
			}

			Set entrySet = htGOtoCount.entrySet();
			Iterator entryIterator = entrySet.iterator();

			int nmatch, nval;
			String szGO;
			while (entryIterator.hasNext()) {
				Map.Entry themap = (Map.Entry) entryIterator.next();
				szGO = (String) themap.getKey();
				nmatch = ((Integer) themap.getValue()).intValue();

				nval = nmatch - 1;
				int ncategoryall = ((Integer) htFullCount.get(szGO)).intValue();

				double dpval = StatUtil.hypergeometrictail(nval, ncategoryall,
						numtotalgenes - ncategoryall, treeptr.numPath);

				Double dminpvalobj = (Double) htGOtoMinOverall.get(szGO);
				if ((dminpvalobj == null)
						|| (dpval < dminpvalobj.doubleValue())) {
					htGOtoMinOverall.put(szGO, new Double(dpval));
				}

				if ((treeptr.parent != null)
						&& (treeptr.parent.numchildren >= 2)) {
					int nparentcategory = ((Integer) htParentGO.get(szGO))
							.intValue();

					double dpvalsplit = StatUtil.hypergeometrictail(nval,
							nparentcategory, treeptr.parent.numPath
									- nparentcategory, treeptr.numPath);

					Double dminpvalsplitobj = (Double) htGOtoMinSplit.get(szGO);
					if ((dminpvalsplitobj == null)
							|| (dpvalsplit < dminpvalsplitobj.doubleValue())) {
						htGOtoMinSplit.put(szGO, new Double(dpvalsplit));
					}
				}
			}

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				findmins(htGOtoMinSplit, htGOtoMinOverall,
						treeptr.nextptr[nchild], genenames, htGOtoCount);
			}
		}
	}

	/**
	 * A record for a GO category include ID, name, p-value based on a split,
	 * and a p-value overall
	 */
	public static class RecIDdrem {
		public String szID;
		public String szName;
		public double dminsplitpval;
		public double dminoverallpval;

		public RecIDdrem(String szID, String szName, double dminoverallpval,
				double dminsplitpval) {
			this.szID = szID;
			this.szName = szName;
			this.dminoverallpval = dminoverallpval;
			this.dminsplitpval = dminsplitpval;
		}
	}

	/**
	 *Compares first by dpvalue (lower comes first) and then by dcategoryselect
	 * (greater comes first)
	 */
	public static class RecIDdremCompare implements Comparator {

		public int compare(Object o1, Object o2) {
			RecIDdrem grec1 = (RecIDdrem) o1;
			RecIDdrem grec2 = (RecIDdrem) o2;
			if (grec1.dminoverallpval < grec2.dminoverallpval) {
				return -1;
			}
			if (grec1.dminoverallpval > grec2.dminoverallpval) {
				return 1;
			}
			if (grec1.dminsplitpval < grec2.dminsplitpval) {
				return -1;
			}
			if (grec1.dminsplitpval > grec2.dminsplitpval) {
				return 1;
			}

			int ncompare = grec1.szName.compareToIgnoreCase(grec2.szName);
			if (ncompare != 0) {
				return ncompare;
			} else {
				return grec1.szID.compareToIgnoreCase(grec2.szID);
			}
		}
	}

}