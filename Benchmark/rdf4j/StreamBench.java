import org.eclipse.rdf4j.model.*;
import java.nio.channels.*;
import org.eclipse.rdf4j.model.impl.*;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.lang.Object;
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
public class StreamBench
{
    public static void main( String[] args ) throws IOException
    {

      	try {
		int FILE_LENGHT = 302924;
                File file =  new File("../rdfStore/str500.txt");
		List<SimpleStatement> statements = new ArrayList<SimpleStatement>();

		FileInputStream fis = new FileInputStream(file);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		ValueFactory factory = SimpleValueFactory.getInstance();

		String line = null;
		while ((line = br.readLine()) != null) {
			String[] parts = line.split(" ");
 			parts[0] = parts[0].substring(1, parts[0].length() - 1);
                        parts[1] = parts[1].substring(1, parts[1].length() - 1);
                        parts[2] = parts[2].substring(1, parts[2].length() - 1);

                        IRI sub = factory.createIRI(parts[0]);
                        IRI pre = factory.createIRI(parts[1]);
			IRI obj = factory.createIRI(parts[2]);
			SimpleStatement statement = (SimpleStatement) factory.createStatement(sub, pre, obj);
			statements.add(statement);
		}

		br.close();

		Iterable<SimpleStatement> iter = statements;

		List timeVec = new LinkedList();
		List timeMemVec = new LinkedList();

		int N_CYCLES = 1;
		List<BindingSet> resultList = null;
		int resultLen = 0;
		for (int i =0; i < N_CYCLES;  i++) {
			System.out.println(i);
			double start = System.nanoTime();
			List<SimpleStatement> currentStm = new LinkedList<SimpleStatement>();
			for (int k = 0; k < statements.size(); k++)
			{
				//System.out.println("value of k is " + k);

				currentStm.add(statements.get(k));

				if (currentStm.size() == 50000  ) {
	        	       		Repository repo = new SailRepository(new MemoryStore());
        		       		repo.initialize();
		        	        RepositoryConnection con = repo.getConnection();

					con.add(currentStm);
					String queryString = "SELECT * WHERE {?s ?p  <http://example.org/int/" + (99) + "> . <http://example.org/int/"  + 0  +  "> ?p ?o }";
					TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
				    	TupleQueryResult result = tupleQuery.evaluate();
				    	resultList = QueryResults.asList(result);

					resultLen += resultList.size();
					currentStm.clear();
					System.out.println(resultLen);
					break;
				}
			}

			double time = System.nanoTime() - start;
			double duration = new Double (time / (double) 1000000);
			timeVec.add(duration);
		}

		double mean = 0;
		double memMean = 0;
		double variance = 0;
		double memVariance = 0;

		for (Object value : timeVec) {
			Double value2 = (Double) value;
			mean += value2.doubleValue();
			variance += value2.doubleValue() * value2.doubleValue();
			System.out.println(value2);
		}

 		mean = mean / (double) timeVec.size();
		variance = variance / (double) (timeVec.size());
		variance = variance - mean * mean;


		System.out.println("Execution Variance is " + variance);
		System.out.println("Execution Mean is " + mean);

		System.out.println("Total lenght is " + resultLen);


	} catch (Exception e) {

		System.out.println("error :");
		System.out.println(e);
		e.printStackTrace();
	}


    }
}
