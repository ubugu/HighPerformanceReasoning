import java.io.*;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import java.util.*;
import uk.ac.ox.cs.JRDFox.store.DataStore;
import uk.ac.ox.cs.JRDFox.store.Parameters;
import uk.ac.ox.cs.JRDFox.store.Resource;
import uk.ac.ox.cs.JRDFox.store.TupleIterator;
import uk.ac.ox.cs.JRDFox.store.DataStore.EqualityAxiomatizationType;
import uk.ac.ox.cs.JRDFox.store.DataStore.Format;
import uk.ac.ox.cs.JRDFox.Prefixes;
import uk.ac.ox.cs.JRDFox.JRDFoxException;
import uk.ac.ox.cs.JRDFox.model.Individual;
import uk.ac.ox.cs.JRDFox.model.GroundTerm;
import mydomain.CircularBuffer;	

public class JRDFoxDemo {

	public static void main(String[] args) throws Exception {
                File file =  new File("../../rdfStore/strts500.txt");

		List<String[]> statements = new ArrayList<String[]>();
		List<Long> timestamps = new ArrayList<Long>();

		//READ FROM FILE
		FileInputStream fis = new FileInputStream(file);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] parts = line.split(" ");
 			parts[0] = parts[0].substring(1, parts[0].length() - 1);
                        parts[1] = parts[1].substring(1, parts[1].length() - 1);
                        parts[2] = parts[2].substring(1, parts[2].length() - 1);

			String[] triple = new String[3];
			triple[0] = parts[0];
			triple[1] = parts[1];
			triple[2] = parts[2];
			statements.add(triple);
			
			if (parts.length==5) {   
				timestamps.add(Long.parseLong(parts[4].substring(0, parts[4].length() - 3)));
			}
		}
		br.close();
		
		int N_CYCLES = 1;

		int totalRes = 0;
		List timeVec = new LinkedList();
		List queryVec = new LinkedList();
		List indexVec = new LinkedList();
		List storeVec = new LinkedList();
		List windowVec = new LinkedList();
		
		
		for (int i =0; i <N_CYCLES; i++)
		{
			double start = System.nanoTime();
			LinkedList<Individual> stmt = new LinkedList<Individual>();	
			int bufferSize = 400000;		
			CircularBuffer<Long> timestampBuffer = new CircularBuffer<Long>(bufferSize, new Long[bufferSize]);
			Long  currentTimestamp = timestamps.get(0) - 1;

			int windowTime = Integer.parseInt(args[1]);
			int stepTime = Integer.parseInt(args[2]);;
			for (int k=0; k <statements.size(); k++)
			{
				timestampBuffer.buffer[k % bufferSize] = timestamps.get(k);
				timestampBuffer.end = k % timestampBuffer.size;

				if ( timestampBuffer.buffer[k % bufferSize] > currentTimestamp +  windowTime) {
					
					double indexStart = System.nanoTime();
					while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
						timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
						stmt.removeFirst();
						stmt.removeFirst();
						stmt.removeFirst();						
					}
					
					double storeStart = System.nanoTime();
					DataStore store = new DataStore(DataStore.StoreType.Sequential);
					Individual[] arrstmt = stmt.toArray(new Individual[stmt.size()]); 
	                        	store.addTriples(arrstmt);
	                        	
	                        	double queryStart = System.nanoTime();
					Prefixes prefixes = Prefixes.DEFAULT_IMMUTABLE_INSTANCE;
					TupleIterator tupleIterator = store.compileQuery(args[0], prefixes, new Parameters());
					totalRes += evaluateAndPrintResults(prefixes, tupleIterator);
					double innerEnd = System.nanoTime();
					
					tupleIterator.dispose();
					store.dispose();
					currentTimestamp += stepTime;
					
					double indexDuration = new Double ((storeStart - indexStart) / (double) 1000000);
                        		double storeDuration = new Double ((queryStart - storeStart) / (double) 1000000);
                        		double queryDuration = new Double ((innerEnd - queryStart) / (double) 1000000);
                        		double totalWindow = new Double((innerEnd - indexStart) / (double) 1000000);
                        		windowVec.add(totalWindow);
                        		queryVec.add(queryDuration);
                        		storeVec.add(storeDuration);
					indexVec.add(indexDuration);
			
				}

				stmt.add(Individual.create(statements.get(k)[0]));
                                stmt.add(Individual.create(statements.get(k)[1]));
                                stmt.add(Individual.create(statements.get(k)[2]));
                        }
			
			if ( stmt.size() != 0) {
					
				while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
					timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
					stmt.removeFirst();
					stmt.removeFirst();
					stmt.removeFirst();					
				}

	                        DataStore store = new DataStore(DataStore.StoreType.Sequential);
                        	store.addTriples(stmt.toArray(new Individual[stmt.size()]));
				Prefixes prefixes = Prefixes.DEFAULT_IMMUTABLE_INSTANCE;
				TupleIterator tupleIterator = store.compileQuery(args[0], prefixes, new Parameters());

				totalRes += evaluateAndPrintResults(prefixes, tupleIterator);

				tupleIterator.dispose();
				store.dispose();
			}

                        double time = System.nanoTime() - start;
                        double duration = new Double (time / (double) 1000000);
                        timeVec.add(duration);
		}
		System.out.println("Number of total results are " + totalRes);




                System.out.println("Index Variance is " + getVariance(indexVec));
                System.out.println("Index Mean is " + getMean(indexVec));
                
                System.out.println("Store creation Variance is " + getVariance(storeVec));
                System.out.println("Store creation Mean is " + getMean(storeVec));

                System.out.println("Query Variance is " + getVariance(queryVec));
                System.out.println("Query Mean is " + getMean(queryVec));

                System.out.println("Window Variance is " + getVariance(windowVec));
                System.out.println("Window Mean is " + getMean(windowVec));

                System.out.println("Execution Variance is " + getVariance(timeVec));
                System.out.println("Execution Mean is " + getMean(timeVec));


	}
	
	public static double getMean(List elements) {
		double mean = 0;
                for (Object value : elements) {
                        Double value2 = (Double) value;
                        mean += value2.doubleValue();
                }
		mean = mean / (double) elements.size();
		return mean;
	}
	
	public static double getVariance(List elements) {
		double mean = 0;
		double variance = 0;
                for (Object value : elements) {
                        Double value2 = (Double) value;
                        mean += value2.doubleValue();
                        variance += value2.doubleValue() * value2.doubleValue();
                }
                
                mean = mean / (double) elements.size();
                variance = variance / (double) (elements.size());
                variance = variance - mean * mean;

		return variance;
	}

	public static int evaluateAndPrintResults(Prefixes prefixes, TupleIterator tupleIterator) throws JRDFoxException {
		int numberOfRows = 0;
		for (long multiplicity = tupleIterator.open(); multiplicity != 0; multiplicity = tupleIterator.advance())  {
			++numberOfRows;
		}
		return numberOfRows;
	}
}
