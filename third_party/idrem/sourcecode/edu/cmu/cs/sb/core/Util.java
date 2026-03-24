package edu.cmu.cs.sb.core;
import java.text.*;
import javax.swing.*;
import java.util.*;
import java.net.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeEvent;


import java.io.*;
import javax.swing.filechooser.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.text.*;
import java.awt.image.*;
import java.text.*;
import java.lang.reflect.*;

/**
 * Class of static utility functions and static variables
 */
public class Util
{
    /**
     * Common chooser object, used across classes so will remember the path
     */
    public static JFileChooser theChooser = new JFileChooser();

    /**
     *Returns a string that has the last period and everything 
     *after it removed.
     */
    public static String stripLastExtension(String szinput)
    {
	String szstriped;
	int nlastindex = szinput.lastIndexOf('.');
	if (nlastindex != -1)
	{
	    szstriped = szinput.substring(0,nlastindex);
	}
	else
	{
	    szstriped = szinput;
	}

	return szstriped;
    }

    /**
     *function coverts roman numeral strings to integers up to 89.
     *assumes the roman numerals are actually properly
     *formed.  Does not look for errors and in case
     *an improper character shows up returns 0.
     */
   public static int romanToNumeric(String roman) 
   {
      int runningSum = 0;
      int carry = 0;

      for (int i = 0; i < roman.length(); i++) 
      {
         switch (roman.charAt(i)) 
         {
	    case 'i': case 'I':
	       runningSum += carry;
	       carry = 1;
	       break;
		
	    case 'v': case 'V':
	       if (carry == 1)
	       {
	          runningSum += 4;
	       }
	       else
	       {
	          runningSum += carry + 5;
	       }
	       carry = 0;
	       break;
		
	    case 'x': case 'X':
	       if (carry == 1) 
               {
	          runningSum += 9;
	          carry = 0;
	       }
	       else 
               {
	          runningSum += carry;
	          carry = 10;
	       }		
	       break;

	    case 'l': case 'L':
	       if (carry == 10)
	       {
	          runningSum += 40;
	       }
	       else
	       {
	          runningSum += carry + 50;
	       }
	       carry = 0;
	       break;
		
	    default:
		return 0;
	 }
      }
	
      return runningSum + carry;
   }


    /**
     *Removes units after last digit
     *Throws an IllegalArgumentException if no digits in szvalunits
     */
    public static double removeUnits(String szvalunits)
    {
	int nindex = 0;
	boolean bfound = false;
	StringBuffer szbuf = new StringBuffer();
	boolean bempty = true;
	while ((nindex < szvalunits.length())&&(!bfound))
	{
	    char ch = szvalunits.charAt(nindex);
	    if ((Character.isDigit(ch))||(ch == '.'))
	    {
		szbuf.append(ch);
		if (ch != '.')
		{
		   bempty = false;
		}
	    }
	    else if (!bempty)
	    {
		bfound = true;
	    }
	    nindex++;
	}

	if (bempty) 
	{
	   throw new IllegalArgumentException("Invalid X-axis label "+szvalunits);
	}
	return Double.parseDouble(szbuf.toString());
    }

    /**
     *Converts double into a formatted string representation
     */
    public static String doubleToSz(double dval)
    {

        String szexp;
        double dtempval = dval;
        int nexp = 0;
	
        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);     
	

        NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);
        nf1.setMinimumFractionDigits(1);
        nf1.setMaximumFractionDigits(1);  

        if (dval <= 0)
	{
	    szexp = "0.00";
	}
        else
	{
           while ((dtempval<0.995)&&(dtempval>0))
	   {
               nexp--;
               dtempval = dtempval*10;
	   }

           if (nexp < -2)
	   {
	      dtempval = Math.pow(10,Math.log(dval)/Math.log(10)-nexp);
	      szexp = nf1.format(dtempval)+"E"+nexp;

	   }
           else
	   {
              szexp = nf2.format(dval);
	   }
	}
        return szexp;
    }

    /**
     *Returns an ImageIcon, or null if the path was invalid. 
     */
    public static ImageIcon createImageIcon(String path) 
    {
        java.net.URL imgURL = Util.class.getResource(path);
        if (imgURL != null) 
        {
            return new ImageIcon(imgURL);
        } 
        else 
        {
            System.err.println("Couldn't find file: " + path);
            return null;
        }
    }


    /**
     * Returns an URL for the path, or null if the path was invalid. 
     */
    public static URL getImageURL(String path) 
    {
        java.net.URL imgURL = Util.class.getResource(path);
        if (imgURL != null) 
        {
            return imgURL;
        } 
        else 
        {
            System.err.println("Couldn't find file: " + path);
            return null;
        }
    }


    /**
     *Calls renderDialog with Help in the title window for a dialog
     */
    public static void renderDialog(JDialog thedialog,String szMessage,int noffsetx, int noffsety)
    {
	renderDialog(thedialog,szMessage,noffsetx,noffsety,"Help");
    }


    /**
     * Calls  renderDialog with Help in the title window for a JFrame
     */
    public static void renderDialog(JFrame theframe,String szMessage,int noffsetx, int noffsety)
    {
	//textArea ==> szMessage
	renderDialog(theframe,szMessage,noffsetx,noffsety,"Help");
    }

    /**
     * Renders a dialog window with szMessage attached to thedialog at location (noffsetx,noffsety)
     * with title szTtitle
     */
    public static void renderDialog(JDialog thedialog,String szMessage,int noffsetx, int noffsety,String szTitle)
    {
	final JDialog thedialogf = thedialog;
	final JTextArea textAreaf  = new JTextArea(szMessage);	
        final int noffsetxf = noffsetx;
        final int noffsetyf = noffsety;
	final String szTitlef = szTitle; 
        final int nlengthf = szMessage.length();
        javax.swing.SwingUtilities.invokeLater(new Runnable() 
        {
           public void run() 
           {
	      JDialog helpDialog = new JDialog(thedialogf, szTitlef, false);
              Container theHelpDialogPane = helpDialog.getContentPane();
       
              helpDialog.setBackground(Color.white);
              theHelpDialogPane.setBackground(Color.white);
       
	      textAreaf.setLineWrap(true);
	      textAreaf.setWrapStyleWord(true);

              textAreaf.setBackground(Color.white);
              textAreaf.setEditable(false);
	      JScrollPane jsp =new JScrollPane(textAreaf);

              theHelpDialogPane.add(jsp);

              helpDialog.setLocation(thedialogf.getX()+noffsetxf,thedialogf.getY()+noffsetyf);

	      if (nlengthf < 600)
	      {      
                 helpDialog.setSize(700,150);
	      }
	      else if (nlengthf < 1000)
	      {      
                 helpDialog.setSize(700,250);
	      }
	      else
	      {      
                 helpDialog.setSize(700,350);
	      }

              helpDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
              helpDialog.setVisible(true); 
	   }
	});
    }

    /////////////////////////////////////////////////////////////////////////

    /**
     * Calls renderDialog with an offset X value of 25 and an offset Y value of 10
     */
    public static void renderDialog(JFrame theframe,String szMessage)
    {
	renderDialog(theframe,szMessage,25,10,"Help");
    }


    //////////////////////////////////////////////////////////////////////////

    /**
     * Renders a dialog window with szMessage attached to theframe at location (noffsetx,noffsety)
     * with title szTtitle
     */
    public static void renderDialog(JFrame theframe,String szMessage,int noffsetx, int noffsety,String szTitle)
    {
	final JFrame theframef = theframe;
	final JTextArea textAreaf  = new JTextArea(szMessage);	
        final int noffsetxf = noffsetx;
        final int noffsetyf = noffsety;
	final String szTitlef = szTitle; 
        final int nlengthf = szMessage.length();
        javax.swing.SwingUtilities.invokeLater(new Runnable() 
        {
           public void run() 
           {
              JDialog helpDialog = new JDialog(theframef, szTitlef, false);
              Container theHelpDialogPane = helpDialog.getContentPane();
       
              helpDialog.setBackground(Color.white);
              theHelpDialogPane.setBackground(Color.white);
       
	      textAreaf.setLineWrap(true);
	      textAreaf.setWrapStyleWord(true);

              textAreaf.setBackground(Color.white);
              textAreaf.setEditable(false);
	      JScrollPane jsp =new JScrollPane(textAreaf);
              theHelpDialogPane.add(jsp);

              helpDialog.setLocation(theframef.getX()+noffsetxf,theframef.getY()+noffsetyf);

	      if (nlengthf < 600)
	      {      
                 helpDialog.setSize(700,150);
	      }
	      else if (nlengthf < 1000)
	      {      
                 helpDialog.setSize(700,250);
	      }
	      else
	      {      
                 helpDialog.setSize(700,350);
	      }

              helpDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
              helpDialog.setVisible(true); 
	   }
	});
    }

    //////////////////////////////////////////////////////////////////////////////

    /**
     * Computes the correlation coefficient on the arrays xvalues and yvalues
     */
    public static double correlation(double[] xvalues, double[] yvalues)
    {

	double dsumx = 0,
	       dsumy = 0,
	       dsumxsq = 0,
	       dsumysq = 0,
	       dsumxy = 0,
	       dvarx,
	       dvary,
	       dcoeff;

        int numvalues = 0;
 
        for (int nindex = 0; nindex < xvalues.length; nindex++)
	{
	   dsumx += xvalues[nindex];
           dsumy += yvalues[nindex];
           dsumxsq += xvalues[nindex]*xvalues[nindex];
           dsumysq += yvalues[nindex]*yvalues[nindex];
           dsumxy  += xvalues[nindex]*yvalues[nindex];
           numvalues++;
	}

        if (numvalues==0)
	{
	    dcoeff = 0;
	}
        else
	{
 
           dvarx = dsumxsq - dsumx*dsumx/numvalues;
           dvary = dsumysq - dsumy*dsumy/numvalues;

           if (dvarx*dvary == 0)
	   {
	      dcoeff = 0;
	   }
           else
	   {
              dcoeff = (dsumxy - dsumx*dsumy/numvalues)/Math.sqrt(dvarx*dvary);
           }
	}
        return dcoeff;
    }



    ////////////////////////////////////////////////////////////////////////////////////
    /**
     *Computes correlation coefficient between xvalues and yvalues only for those that
     *have non-zero includex and includey values
     *Returns 0 if no values are agree on for all data
     */
    public static double correlation(double[] xvalues, double[] yvalues,int[] include)
    {
	double dsumx = 0,
	       dsumy = 0,
	       dsumxsq = 0,
	       dsumysq = 0,
	       dsumxy = 0,
	       dvarx,
	       dvary,
	       dcoeff;

        int numvalues = 0;
 
        for (int nindex = 0; nindex < include.length; nindex++)
	{
	    if (include[nindex] != 0)
	    {
		dsumx += xvalues[nindex];
                dsumy += yvalues[nindex];
                dsumxsq += xvalues[nindex]*xvalues[nindex];
                dsumysq += yvalues[nindex]*yvalues[nindex];
                dsumxy  += xvalues[nindex]*yvalues[nindex];
                numvalues++;
	    }
	}

        if (numvalues==0)
	{
	    dcoeff = 0;
	}
        else
	{            
           dvarx = dsumxsq - dsumx*dsumx/numvalues;
           dvary = dsumysq - dsumy*dsumy/numvalues;

           if (dvarx*dvary== 0)
	   {
	      dcoeff = 0;
	   }
           else
	   {
              dcoeff = (dsumxy - dsumx*dsumy/numvalues)/Math.sqrt(dvarx*dvary);
           }
	}
         return dcoeff;
    }


    ////////////////////////////////////////////////////////////////////////////////
    /**
     *Computes correlation coefficient between xvalues and yvalues only for those that
     *have non-zero includex and includey values
     *Returns 0 if no values are agree on for all data
     */
    public static double correlation(double[] xvalues, double[] yvalues,int[] includex,int[] includey)
    {
	double dsumx = 0,
	       dsumy = 0,
	       dsumxsq = 0,
	       dsumysq = 0,
	       dsumxy = 0,
	       dvarx,
	       dvary,
	       dcoeff;

        int numvalues = 0;
 
        for (int nindex = 0; nindex < includex.length; nindex++)
	{
	    if ((includex[nindex] != 0)&&(includey[nindex]!=0))
	    {
		dsumx += xvalues[nindex];
                dsumy += yvalues[nindex];
                dsumxsq += xvalues[nindex]*xvalues[nindex];
                dsumysq += yvalues[nindex]*yvalues[nindex];
                dsumxy  += xvalues[nindex]*yvalues[nindex];
                numvalues++;
	    }
	}
 
        if (numvalues==0)
	{
	    dcoeff = 0;
	}
        else
	{
           dvarx = dsumxsq - dsumx*dsumx/numvalues;
           dvary = dsumysq - dsumy*dsumy/numvalues;

           if (dvarx*dvary==0)
	   {
	      dcoeff = 0;
	   }
           else
	   {
               dcoeff = (dsumxy - dsumx*dsumy/numvalues)/Math.sqrt(dvarx*dvary);
           }
	}
         return dcoeff;
    }



    /////////////////////////////////////////////////////////////////////////////////
    /**
     * Returns the euclidean distance between xvalues and yvalues for those include values
     * that are non-zero.
     */
    public static double euclidean(double[] xvalues, double[] yvalues,int[] include)
    {
	double ddiff;
	double ddist;
	double dsumsq = 0;
 
        for (int nindex = 0; nindex < include.length; nindex++)
	{
	    if (include[nindex] != 0)
	    {
		ddiff = (xvalues[nindex] - yvalues[nindex]);
		dsumsq += ddiff * ddiff;
	    }
	}

	ddist = Math.sqrt(dsumsq);

        return ddist;
    }

    ////////////////////////////////////////////////////////////////////////////////
    /**
     * Returns the sum of the squared difference of xvalues and yvalues for non-zero 
     * include index values
     */
    public static double distortion(double[] xvalues, double[] yvalues,int[] include)
    {
	double ddiff;
	double dsumsq = 0;
 
        for (int nindex = 0; nindex < include.length; nindex++)
	{
	    if (include[nindex] != 0)
	    {
		ddiff = (xvalues[nindex] - yvalues[nindex]);
		dsumsq += ddiff * ddiff;
	    }
	}

        return dsumsq;
    }

    //////////////////////////////////////////////////////////////////////////////////
    /**
     * Returns the median value in vals in indicies between 0 and nvalindex-1
     */
    public static double getmedian(double[] vals,int nvalindex)
    {                      
       double dmedian; 
       Arrays.sort(vals, 0, nvalindex);      
       if (nvalindex % 2 == 0)
       {
          dmedian = (vals[nvalindex/2-1]+vals[nvalindex/2])/2;
       }
       else
       {
          dmedian = vals[nvalindex/2];
       } 

       return dmedian;
    }

}
