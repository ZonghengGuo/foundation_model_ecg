/************************************************************************
 This file contains functions for detecting QRS complexes in an ECG.  The
 QRS detector requires filter functions in qrsfilt.cpp and parameter
 definitions in qrsdet.h.

 This is a MEX-file for MATLAB.
 
 To compile type 'mex eplim.cpp QRSFILT.cpp' in MATLAB command window.
 
 Syntax: QRS=eplim(ECG)
*************************************************************************/

#include <memory.h> /* For memmov. */
#include <string.h>	
#include <math.h>
#include "qrsdet.h"
#include "mex.h"
#define PRE_BLANK	MS200

// External Prototypes.

int QRSFilter(double datum, int init) ;
int deriv1( double x0, int init ) ;

// Local Prototypes.

int Peak( int datum, int init ) ;
int median(int *array, int datnum) ;
int thresh(int qmedian, int nmedian) ;
int BLSCheck(int *dBuf,int dbPtr,int *maxder) ;

int earlyThresh(int qmedian, int nmedian) ;


double TH = 0.475  ;

int DDBuffer[DER_DELAY], DDPtr ;	/* Buffer holding derivative data. */
int Dly  = 0 ;

const int MEMMOVELEN = 7*sizeof(int);

int QRSDet( double datum, int init )
	{
	static int det_thresh, qpkcnt = 0 ;
	static int qrsbuf[8], noise[8], rrbuf[8] ;
	static int rsetBuff[8], rsetCount = 0 ;
	static int nmedian, qmedian, rrmedian ;
	static int count, sbpeak = 0, sbloc, sbcount = MS1500 ;
	static int maxder, lastmax ;
	static int initBlank, initMax ;
	static int preBlankCnt, tempPeak ;
	
	int fdatum, QrsDelay = 0 ;
	int i, newPeak, aPeak ;

/*	Initialize all buffers to 0 on the first call.	*/

	if( init )
		{
		for(i = 0; i < 8; ++i)
			{
			noise[i] = 0 ;	/* Initialize noise buffer */
			rrbuf[i] = MS1000 ;/* and R-to-R interval buffer. */
			}

		qpkcnt = maxder = lastmax = count = sbpeak = 0 ;
		initBlank = initMax = preBlankCnt = DDPtr = 0 ;
		sbcount = MS1500 ;
		QRSFilter(0,1) ;	/* initialize filters. */
		Peak(0,1) ;
		}

	fdatum = QRSFilter(datum,0) ;	/* Filter data. */


	/* Wait until normal detector is ready before calling early detections. */

	aPeak = Peak(fdatum,0) ;

	// Hold any peak that is detected for 200 ms
	// in case a bigger one comes along.  There
	// can only be one QRS complex in any 200 ms window.

	newPeak = 0 ;
	if(aPeak && !preBlankCnt)			// If there has been no peak for 200 ms
		{										// save this one and start counting.
		tempPeak = aPeak ;
		preBlankCnt = PRE_BLANK ;			// MS200
		}

	else if(!aPeak && preBlankCnt)	// If we have held onto a peak for
		{										// 200 ms pass it on for evaluation.
		if(--preBlankCnt == 0)
			newPeak = tempPeak ;
		}

	else if(aPeak)							// If we were holding a peak, but
		{										// this ones bigger, save it and
		if(aPeak > tempPeak)				// start counting to 200 ms again.
			{
			tempPeak = aPeak ;
			preBlankCnt = PRE_BLANK ; // MS200
			}
		else if(--preBlankCnt == 0)
			newPeak = tempPeak ;
		}

/*	newPeak = 0 ;
	if((aPeak != 0) && (preBlankCnt == 0))
		newPeak = aPeak ;
	else if(preBlankCnt != 0) --preBlankCnt ; */



	/* Save derivative of raw signal for T-wave and baseline
	   shift discrimination. */
	
	DDBuffer[DDPtr] = deriv1( datum, 0 ) ;
	if(++DDPtr == DER_DELAY)
		DDPtr = 0 ;

	/* Initialize the qrs peak buffer with the first eight 	*/
	/* local maximum peaks detected.						*/

	if( qpkcnt < 8 )
		{
		++count ;
		if(newPeak > 0) count = WINDOW_WIDTH ;
		if(++initBlank == MS1000)
			{
			initBlank = 0 ;
			qrsbuf[qpkcnt] = initMax ;
			initMax = 0 ;
			++qpkcnt ;
			if(qpkcnt == 8)
				{
				qmedian = median( qrsbuf, 8 ) ;
				nmedian = 0 ;
				rrmedian = MS1000 ;
				sbcount = MS1500+MS150 ;
				det_thresh = thresh(qmedian,nmedian) ;
				}
			}
		if( newPeak > initMax )
			initMax = newPeak ;
		}

	else	/* Else test for a qrs. */
		{
		++count ;
		if(newPeak > 0)
			{
			
			
			/* Check for maximum derivative and matching minima and maxima
			   for T-wave and baseline shift rejection.  Only consider this
			   peak if it doesn't seem to be a base line shift. */
			   
			if(!BLSCheck(DDBuffer, DDPtr, &maxder))
				{


				// Classify the beat as a QRS complex
				// if the peak is larger than the detection threshold.

				if(newPeak > det_thresh)
					{
					memmove(&qrsbuf[1], qrsbuf, MEMMOVELEN) ;
					qrsbuf[0] = newPeak ;
					qmedian = median(qrsbuf,8) ;
					det_thresh = thresh(qmedian,nmedian) ;
					memmove(&rrbuf[1], rrbuf, MEMMOVELEN) ;
					rrbuf[0] = count - WINDOW_WIDTH ;
					rrmedian = median(rrbuf,8) ;
					sbcount = rrmedian + (rrmedian >> 1) + WINDOW_WIDTH ;
					count = WINDOW_WIDTH ;

					sbpeak = 0 ;

					lastmax = maxder ;
					maxder = 0 ;
					QrsDelay =  WINDOW_WIDTH + FILTER_DELAY ;
					initBlank = initMax = rsetCount = 0 ;

			//		preBlankCnt = PRE_BLANK ;
					}

				// If a peak isn't a QRS update noise buffer and estimate.
				// Store the peak for possible search back.


				else
					{
					memmove(&noise[1],noise,MEMMOVELEN) ;
					noise[0] = newPeak ;
					nmedian = median(noise,8) ;
					det_thresh = thresh(qmedian,nmedian) ;

					// Don't include early peaks (which might be T-waves)
					// in the search back process.  A T-wave can mask
					// a small following QRS.

					if((newPeak > sbpeak) && ((count-WINDOW_WIDTH) >= MS360))
						{
						sbpeak = newPeak ;
						sbloc = count  - WINDOW_WIDTH ;
						}
					}
				}
			}
		
		/* Test for search back condition.  If a QRS is found in  */
		/* search back update the QRS buffer and det_thresh.      */

		if((count > sbcount) && (sbpeak > (det_thresh >> 1)))
			{
			memmove(&qrsbuf[1],qrsbuf,MEMMOVELEN) ;
			qrsbuf[0] = sbpeak ;
			qmedian = median(qrsbuf,8) ;
			det_thresh = thresh(qmedian,nmedian) ;
			memmove(&rrbuf[1],rrbuf,MEMMOVELEN) ;
			rrbuf[0] = sbloc ;
			rrmedian = median(rrbuf,8) ;
			sbcount = rrmedian + (rrmedian >> 1) + WINDOW_WIDTH ;
			QrsDelay = count = count - sbloc ;
			QrsDelay += FILTER_DELAY ;
			sbpeak = 0 ;
			lastmax = maxder ;
			maxder = 0 ;
			initBlank = initMax = rsetCount = 0 ;
			}
		}

	// In the background estimate threshold to replace adaptive threshold
	// if eight seconds elapses without a QRS detection.

	if( qpkcnt == 8 )
		{
		if(++initBlank == MS1000)
			{
			initBlank = 0 ;
			rsetBuff[rsetCount] = initMax ;
			initMax = 0 ;
			++rsetCount ;

			// Reset threshold if it has been 8 seconds without
			// a detection.

			if(rsetCount == 8)
				{
				for(i = 0; i < 8; ++i)
					{
					qrsbuf[i] = rsetBuff[i] ;
					noise[i] = 0 ;
					}
				qmedian = median( rsetBuff, 8 ) ;
				nmedian = 0 ;
				rrmedian = MS1000 ;
				sbcount = MS1500+MS150 ;
				det_thresh = thresh(qmedian,nmedian) ;
				initBlank = initMax = rsetCount = 0 ;
            sbpeak = 0 ;
				}
			}
		if( newPeak > initMax )
			initMax = newPeak ;
		}

	return(QrsDelay) ;
	}

/**************************************************************
* peak() takes a datum as input and returns a peak height
* when the signal returns to half its peak height, or 
**************************************************************/

int Peak( int datum, int init )
	{
	static int max = 0, timeSinceMax = 0, lastDatum ;
	int pk = 0 ;

	if(init)
		max = timeSinceMax = 0 ;
		
	if(timeSinceMax > 0)
		++timeSinceMax ;

	if((datum > lastDatum) && (datum > max))
		{
		max = datum ;
		if(max > 2)
			timeSinceMax = 1 ;
		}

	else if(datum < (max >> 1))
		{
		pk = max ;
		max = 0 ;
		timeSinceMax = 0 ;
		Dly = 0 ;
		}

	else if(timeSinceMax > MS95)
		{
		pk = max ;
		max = 0 ;
		timeSinceMax = 0 ;
		Dly = 3 ;
		}
	lastDatum = datum ;
	return(pk) ;
	}

/********************************************************************
median returns the median of an array of integers.  It uses a slow
sort algorithm, but these arrays are small, so it hardly matters.
********************************************************************/

int median(int *array, int datnum)
	{
	int i, j, k, temp, sort[20] ;
	for(i = 0; i < datnum; ++i)
		sort[i] = array[i] ;
	for(i = 0; i < datnum; ++i)
		{
		temp = sort[i] ;
		for(j = 0; (temp < sort[j]) && (j < i) ; ++j) ;
		for(k = i - 1 ; k >= j ; --k)
			sort[k+1] = sort[k] ;
		sort[j] = temp ;
		}
	return(sort[datnum>>1]) ;
	}
/*
int median(int *array, int datnum)
	{
	long sum ;
	int i ;

	for(i = 0, sum = 0; i < datnum; ++i)
		sum += array[i] ;
	sum /= datnum ;
	return(sum) ;
	} */

/****************************************************************************
 thresh() calculates the detection threshold from the qrs median and noise
 median estimates.
****************************************************************************/

int thresh(int qmedian, int nmedian)
	{
	int thrsh, dmed ;
	double temp ;
	dmed = qmedian - nmedian ;
/*	thrsh = nmedian + (dmed>>2) + (dmed>>3) + (dmed>>4); */
	temp = dmed ;
	temp *= TH ;
	dmed = temp ;
	thrsh = nmedian + dmed ; /* dmed * THRESHOLD */
	return(thrsh) ;
	}

/***********************************************************************
	BLSCheck() reviews data to see if a baseline shift has occurred.
	This is done by looking for both positive and negative slopes of
	roughly the same magnitude in a 220 ms window.
***********************************************************************/

int BLSCheck(int *dBuf,int dbPtr,int *maxder)
	{
	int max, min, maxt, mint, t, x ;
	max = min = 0 ;

	return(0) ;
	
	for(t = 0; t < MS220; ++t)
		{
		x = dBuf[dbPtr] ;
		if(x > max)
			{
			maxt = t ;
			max = x ;
			}
		else if(x < min)
			{
			mint = t ;
			min = x;
			}
		if(++dbPtr == DER_DELAY)
			dbPtr = 0 ;
		}

	*maxder = max ;
	min = -min ;
	
	/* Possible beat if a maximum and minimum pair are found
		where the interval between them is less than 150 ms. */
	   
	if((max > (min>>3)) && (min > (max>>3)) &&
		(abs(maxt - mint) < MS150))
		return(0) ;
		
	else
		return(1) ;
	}



void mexFunction (int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
    double *ecg;                    /*Pointer to double for input data*/
    double *outArray;               /*Pointer to double for output data*/
    void *dyn;                      /*Pointer to void for the dynamic allocation of memory (mCalloc)*/
    int N=0;
    int i=1;
    int j=0; 
    int k=0;
    /*CHECK FOR PROPER NUMBER OF ARGUMENTS*/
    
    if (nrhs != 1 ) mexErrMsgIdAndTxt("EplimitedQRSDetector:NoInput", "This function takes one input argument: ECG.");
    else if(nlhs!=1) mexErrMsgIdAndTxt("EplimitedQRSDetector:NoOutput", "This function requires one output argument.");
    
    
    
    /*LOAD INPUT DATA AND ALLOCATE OUTPUT MEMORY*/
    ecg=mxGetPr(prhs[0]);                       /*Input data loading*/
    N=(int) mxGetM(prhs[0]);                    /*Data array length*/  
    plhs[0]=mxCreateDoubleMatrix(0,0,mxREAL);   /*Initialize a 0x0 output matrix. Necessary memory will be dinamically allocated*/
    dyn = mxCalloc(N,sizeof(double));           /*Dynamic memory allocation*/
    outArray=(double*) dyn;                     /*Assign pointer-to-void to a pointer-to-double. outArray now points the dynamic memory.*/
                                              
    
    
    /*CALL THE SUBROUTINE*/
    
    for (j=0;j<N;j++){
        outArray[k]=QRSDet(ecg[j], i );         /*QRSDet output: Detection delay (it waits a variable time window to change the detection if others peaks are found)*/
        if (outArray[k]!=0){ 
            outArray[k]=j-outArray[k];          /*If a QRS is detected, its location is calculated subtracting detection delay from current sample, and stored.*/
            k++;
        }
        i=0;
    }
   
   
    /*FILL THE OUTPUT ARRAY*/
    mxSetData(plhs[0], outArray);               /*Assign detection data contained in outArray to output matrix plhs[0]*/
    mxSetM(plhs[0], N);                         /*Assign Nx1 dimensions to plhs[0]*/
    mxSetN(plhs[0], 1);
    return;
}