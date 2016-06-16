import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.util.*;

import org.eclipse.rdf4j.model.Value;
import org.eclipse.rdf4j.model.ValueFactory;
import org.eclipse.rdf4j.query.BindingSet;
import org.eclipse.rdf4j.query.QueryLanguage;
import org.eclipse.rdf4j.query.QueryResults;
import org.eclipse.rdf4j.query.TupleQuery;
import org.eclipse.rdf4j.query.TupleQueryResult;
import org.eclipse.rdf4j.repository.Repository;
import org.eclipse.rdf4j.repository.RepositoryConnection;
import org.eclipse.rdf4j.repository.sail.SailRepository;
import org.eclipse.rdf4j.rio.RDFFormat;
import org.eclipse.rdf4j.sail.memory.MemoryStore;


/**
 * Hello world!
 *
 */
public class App2
{
    public static void main( String[] args ) throws IOException
    {
        
//        File file =  new File("../rdfStore/rdf2.txt");
        
        //copy();
        try {
				
		System.out.println("printo");
		int N_CYCLES = 100;
		List timeVec = new LinkedList();
		double norm = 1000000;
		try {
			
			Repository repo = null;
			RepositoryConnection con = null;

			for (int i =0; i <N_CYCLES; i++) 
			{
		        File file =  new File("../rdfStore/rdf2.txt");


			repo = new SailRepository(new MemoryStore());
			repo.initialize();
			con = repo.getConnection();
			con.add(file, "", RDFFormat.NTRIPLES);
			System.out.println("start time");
		        double startTime = System.nanoTime();
			String queryString = "SELECT ?p ?y WHERE { <http://example.org/int/0> ?p ?y . <http://example.org/int/0> ?p ?y } ";
			TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
		    	List<BindingSet> resultList;
		    	TupleQueryResult result = tupleQuery.evaluate();
		    	resultList = QueryResults.asList(result);
		    	double endTime = System.nanoTime();
			double duration = (endTime - startTime) / norm;
			timeVec.add(duration);
		    	System.out.println(duration);
			}

			double mean = 0;
			double variance = 0;
			for (Object value : timeVec) {
				Double casted = (Double) value;
				mean += casted.doubleValue();
				variance += casted.doubleValue() * casted.doubleValue();
			}
			mean = mean /  (double) timeVec.size();
			variance = variance / (double) timeVec.size();
			variance = variance - mean * mean;
			
			System.out.println("mean is " + mean);
			System.out.println("varaince is " + variance);
		       
	   	}   catch (Exception e) {
	   		
	   		System.out.println("inner error");
	   		System.out.println(e);      
	      	}
	     
	   	
	   	
	} catch (Exception e) {
	
		System.out.println("outer error");
		e.printStackTrace();
	
	}
        
        System.out.println("Funza");
    }
    
    
    public static void copy() throws IOException
    {
        File file =  new File("C:/Users/Claudio/Downloads/eclipse-rdf4j-2.0M2-sdk/rdf.txt");
        
        
			String baseURI = "http://example.org/int/";
			String sCurrentLine;
			BufferedReader br;
			br = new BufferedReader(new FileReader("C:/Users/Claudio/Downloads/eclipse-rdf4j-2.0M2-sdk/rdf.txt"));
			PrintWriter writer = new PrintWriter("C:/Users/Claudio/Downloads/eclipse-rdf4j-2.0M2-sdk/rdf2.txt", "UTF-8");
			int index = 1;
			while ((sCurrentLine = br.readLine()) != null) {
				String[] splitted = sCurrentLine.split(" ");
				splitted[0] = "<" + baseURI + splitted[0] + ">";
				splitted[1] = "<" + baseURI + splitted[1] + ">";
				splitted[2] = "<" + baseURI + splitted[2] + ">";
				if (index % 77 == 0) {
					splitted[0] = "<" + baseURI + "0" + ">";
				}
				writer.println(splitted[0] + " " +  splitted[1] + " " +  splitted[2] + " " +  splitted[3]);
				index++;
			}
			
			try {
			      //con.add(file, baseURI, RDFFormat.NTRIPLES);
			      URL url = new URL("http://example.org/example/remote.rdf");
			     
			 
		   }
	       
	       catch (Exception e) {
	        	   // handle exception
	       }
	           
	

		
        
    }
}
