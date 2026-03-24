package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import java.io.*;
import java.util.*;
import javax.swing.filechooser.FileFilter;

/**
 * Class used to specify the file filtering in the file chooser menu. This class
 * is used by DREMGui_SaveDREM
 */
public class DREMGui_ImageFilter extends FileFilter {

	String szdesc;
	HashSet validExt = new HashSet();

	/**
	 * Returns the extension of the file
	 */
	public static String getExtension(File f) {
		String szext = "";
		String szname = f.getName();
		int nindex = szname.lastIndexOf('.');
		if (nindex > 0 && nindex < szname.length() - 1) {
			szext = szname.substring(nindex + 1).toLowerCase(Locale.ENGLISH);
		}
		return szext;
	}

	/**
	 * Returns true iff f is not null and a directory or has a non-null
	 * extension in validExt
	 */
	public boolean accept(File f) {
		if (f != null) {
			if (f.isDirectory()) {
				return true;
			}

			String extension = getExtension(f);
			if (extension != null && validExt.contains(getExtension(f))) {
				return true;
			}
			;
		}
		return false;
	}

	/**
	 * Returns szdesc
	 */
	public String getDescription() {
		return szdesc;
	}

	/**
	 * Sets the value of szdesc
	 */
	public void setDescription(String szdesc) {
		this.szdesc = szdesc;
	}

	/**
	 * Adds szext into validExt
	 */
	public void addExtension(String szext) {
		validExt.add(szext);
	}
}