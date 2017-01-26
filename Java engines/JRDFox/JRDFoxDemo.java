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
import uk.ac.ox.cs.JRDFox.model.Literal;
import uk.ac.ox.cs.JRDFox.model.Datatype;
import mydomain.CircularBuffer;	
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.file.Files;

public class JRDFoxDemo {

	public static void main(String[] args) throws Exception {
                File file =  new File(args[3]);
		
		List<String[]> statements = new ArrayList<String[]>();
		List<Long> timestamps = new ArrayList<Long>();

		//READ FROM FILE
		FileInputStream fis = new FileInputStream(file);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		String line = null;
		int value = 0;
		while ((line = br.readLine()) != null) {
			//String[] parts = line.split(" ",3);
			String[] parts = line.split(" ");
 			parts[0] = parts[0].substring(1, parts[0].length() - 1);
                        parts[1] = parts[1].substring(1, parts[1].length() - 1);
                        //parts[2] = parts[2].substring(1, parts[2].length() - 1);
                        
                       /* parts[2] = parts[2].substring(0, parts[2].length() - 2); */
                        if (parts[2].charAt(0) == '<') {
                        	parts[2] = parts[2].substring(1, parts[2].length() - 1);
                        } 
                        
                        
			String[] triple = new String[3];
			triple[0] = parts[0];
			triple[1] = parts[1];
			triple[2] = parts[2];
			statements.add(triple);
			
			if (parts.length>=5) {   
				timestamps.add(Long.parseLong(parts[4].substring(0, parts[4].length() - 3)));
				//timestamps.add(new Long(12600));
			}
		}
		br.close();


			
		int N_CYCLES = 1;

		int TEST_CYCLES = 1;

		int totalRes = 0;
		List timeVec = new LinkedList();
		List queryVec = new LinkedList();
		List qqueryVec = new LinkedList();
		List indexVec = new LinkedList();
		List storeVec = new LinkedList();
		List windowVec = new LinkedList();
		
		System.out.println(timestamps.size() + " " + statements.size() );
		
		for (int i =0; i <N_CYCLES; i++)
		{
			int time = 0;
			boolean isLaunched = false;
			LinkedList<GroundTerm> stmt = new LinkedList<GroundTerm>();	
			int bufferSize = 400000;		
			CircularBuffer<Long> timestampBuffer = new CircularBuffer<Long>(bufferSize, new Long[bufferSize]);
			Long  currentTimestamp = timestamps.get(0) - 1;

			int windowTime = Integer.parseInt(args[1]);
			int stepTime = Integer.parseInt(args[2]);

			
			int k  =0;
		/*	for (; ((k <statements.size()) && (!isLaunched)); k++)
			{
				timestampBuffer.buffer[k % bufferSize] = timestamps.get(k);
				timestampBuffer.end = k % timestampBuffer.size;

				if ( timestampBuffer.buffer[k % bufferSize] > currentTimestamp +  windowTime) {
					

					while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
						timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
						stmt.removeFirst();
						stmt.removeFirst();
						stmt.removeFirst();						
					}
			
					DataStore store = new DataStore(DataStore.StoreType.Sequential);
					GroundTerm[] arrstmt = stmt.toArray(new GroundTerm[stmt.size()]); 
	                        	store.addTriples(arrstmt);
	                        	

					Prefixes prefixes = Prefixes.DEFAULT_IMMUTABLE_INSTANCE;
					TupleIterator tupleIterator = store.compileQuery(args[0], prefixes, new Parameters());
					totalRes += evaluateAndPrintResults(prefixes, tupleIterator);;
					
					tupleIterator.dispose();
					store.dispose();
					currentTimestamp += stepTime;
					
				/*	double indexDuration = new Double ((storeStart - indexStart) / (double) 1000000);
                        		double storeDuration = new Double ((queryStart - storeStart) / (double) 1000000);
                        		double queryDuration = new Double ((innerEnd - queryStart) / (double) 1000000);
                        		double totalWindow = new Double((innerEnd - indexStart) / (double) 1000000);
                        		windowVec.add(totalWindow);
                        		queryVec.add(queryDuration);
                        		storeVec.add(storeDuration);
					indexVec.add(indexDuration);
					
					isLaunched = true;
			
				}

				stmt.add(Individual.create(statements.get(k)[0]));
                                stmt.add(Individual.create(statements.get(k)[1]));
                                
                                if (statements.get(k)[2].charAt(0) != '"') {
                                	stmt.add(Individual.create(statements.get(k)[2]));
                                } else  {
                                	Literal lit =  createLiteral(statements.get(k)[2]);
                                	stmt.add(lit);
                                }
                        }*/


                        
                        System.out.println("launched at  " + k);
			
			
double queryStart = System.nanoTime();                      
			for (; k <statements.size(); k++)
			{
				timestampBuffer.buffer[k % bufferSize] = timestamps.get(k);
				timestampBuffer.end = k % timestampBuffer.size;

				if ( timestampBuffer.buffer[k % bufferSize] > currentTimestamp +  windowTime) {
					

					while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
						timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
						stmt.removeFirst();
						stmt.removeFirst();
						stmt.removeFirst();						
					}
			
					DataStore store = new DataStore(DataStore.StoreType.Sequential);
					GroundTerm[] arrstmt = stmt.toArray(new GroundTerm[stmt.size()]); 
	                        	store.addTriples(arrstmt);
	                        	

					Prefixes prefixes = Prefixes.DEFAULT_IMMUTABLE_INSTANCE;
					TupleIterator tupleIterator = store.compileQuery(args[0], prefixes, new Parameters());
					totalRes += evaluateAndPrintResults(prefixes, tupleIterator);;
					


					tupleIterator.dispose();
					store.dispose();
					currentTimestamp += stepTime;
					

					 time++;
			
			
				}

				stmt.add(Individual.create(statements.get(k)[0]));
                                stmt.add(Individual.create(statements.get(k)[1]));
                                
                                if (statements.get(k)[2].charAt(0) != '"') {
                                	stmt.add(Individual.create(statements.get(k)[2]));
                                } else  {
                                	Literal lit =  createLiteral(statements.get(k)[2]);
                                	stmt.add(lit);
                                }
                                
                               
                        }


  

			if ( stmt.size() != 0) {
			
				while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
					timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
					stmt.removeFirst();
					stmt.removeFirst();
					stmt.removeFirst();					
				}
				
				

	                        DataStore store = new DataStore(DataStore.StoreType.Sequential);
                        	store.addTriples(stmt.toArray(new GroundTerm[stmt.size()]));
                        	TupleIterator tupleIterator = null;

double qqueryStart = System.nanoTime();  			                        	
				Prefixes prefixes = Prefixes.DEFAULT_IMMUTABLE_INSTANCE;
				tupleIterator = store.compileQuery(args[0], prefixes, new Parameters());
				totalRes += evaluateAndPrintResults(prefixes, tupleIterator);				
double iinnerEnd = System.nanoTime();
double qqueryDuration = new Double ((iinnerEnd - qqueryStart) / (double) 1000000);                       	
qqueryVec.add(qqueryDuration); 				
				tupleIterator.dispose();
				store.dispose();
				 time++;
				 		 System.out.println("LAUNCHED --");
				
			}

double innerEnd = System.nanoTime();
double queryDuration = new Double ((innerEnd - queryStart) / (double) 1000000);                       	
queryVec.add(queryDuration); 

		System.out.println("TIMES " + time);
                        
		}
		

		queryVec.remove(0);
		qqueryVec.remove(0);
		
		Double mean = new Double(getMean(queryVec));
		Double mean2 = new Double(getMean(qqueryVec));
		
		for (int i = 0; i < queryVec.size(); i++) {
			System.out.println(queryVec.get(i));
		}
		
		
                System.out.println("total res are " + totalRes);
                
                
		Files.write(Paths.get("./result.txt"),  (mean2.toString() + " \n").getBytes(), StandardOpenOption.APPEND);

	}
	
	
	//Function for creating literal, deducing the type from the string
	static Literal createLiteral(String literalStr) {
		int length = literalStr.length();
   
                                 	
                //SENZA VIRGOLETTE
		if (literalStr.charAt(length - 1) == '"') {
			String litvalue = literalStr.substring(1, length - 1);
			Literal literal = Literal.create(litvalue, Datatype.XSD_STRING);
			return literal;
		}

		if (literalStr.charAt(length - 4) == '"') {
			String litvalue = literalStr.substring(1, length - 4);
			Literal literal = Literal.create(litvalue, Datatype.XSD_STRING);
			return literal;
		}
		
		if  (literalStr.charAt(length - 2) == 'g') {
			String litvalue = literalStr.substring(1, length - 43);
			Literal literal = Literal.create(litvalue, Datatype.XSD_STRING);
			return literal;
		}

		if (literalStr.charAt(length - 2) == 'r') {
			String litvalue = literalStr.substring(1, length - 45);
			Literal literal = Literal.create(litvalue, Datatype.XSD_INTEGER);
			return literal;
		}

	
		if (literalStr.charAt(length - 2) == 't') {
			String litvalue = literalStr.substring(1, length - 43);
			Literal literal = Literal.create(litvalue, Datatype.XSD_FLOAT);
			return literal;
		}	

		if (literalStr.charAt(length - 3) == 'S') {
			String litvalue = literalStr.substring(1, length - 66);
			litvalue = litvalue.replace('.',',');
			Literal literal = Literal.create(litvalue, Datatype.XSD_DECIMAL);
			return literal;
		}	
	
		if (literalStr.charAt(length - 3) == 'l') {
			String litvalue = literalStr.substring(1, length - 44);
			Literal literal = Literal.create(litvalue, Datatype.XSD_DOUBLE);
			System.out.println(literal);
			return literal;
		}

		if (literalStr.charAt(length - 3) == 't') {
			String litvalue = literalStr.substring(1, length - 42);
			Literal literal = Literal.create(litvalue, Datatype.XSD_DATE);
			return literal;
		}

		if (literalStr.charAt(length - 3) == 'm') {
			String litvalue = literalStr.substring(1, length - 46);
			Literal literal = Literal.create(litvalue, Datatype.XSD_DATE_TIME);
			return literal;
		}
		



		System.out.println("ERROR");
		System.out.println( "--" + literalStr + "- 2 " + literalStr.charAt(length - 1));
		return null;
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
	
	/*public static int  evaluateAndPrintResults(Prefixes prefixes, TupleIterator tupleIterator) throws JRDFoxException {
		int numberOfRows = 0;
		System.out.println();
		System.out.println("=======================================================================================");
		int arity = tupleIterator.getArity();
		// We iterate trough the result tuples
		for (long multiplicity = tupleIterator.open(); multiplicity != 0; multiplicity = tupleIterator.advance()) {
			// We iterate trough the terms of each tuple
			for (int termIndex = 0; termIndex < arity; ++termIndex) {
				if (termIndex != 0)
					System.out.print("  ");
				// For each term we get a Resource object that contains the lexical form and the data type of the term.
				// One can also access terms as GroundTerm objects from the uk.ac.ox.cs.JRDFox.model package using the 
				// method TupleIterator.getGroundTerm(int termIndex). Using objects form the uk.ac.ox.cs.JRDFox.model 
				// package has the benefit of ensuring that at any point each term is represented by at most one Java 
				// object. This benefit, however, comes at a price, since, unlike in the case of Resource objects, the 
				// creation of GroundTerm objects involves a hashtable lookup, which in some cases can lead to a 
				// significant overhead. 
				Resource resource = tupleIterator.getResource(termIndex);
				System.out.print(resource.toString(prefixes));
			}
			System.out.print(" * ");
			System.out.print(multiplicity);
			System.out.println();
			++numberOfRows;
		}
		System.out.println("---------------------------------------------------------------------------------------");
		System.out.println("  The number of rows returned: " + numberOfRows);
		System.out.println("=======================================================================================");
		System.out.println();
		return numberOfRows;
	}*/
	
}
