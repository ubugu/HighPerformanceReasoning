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

public class JRDFoxDemo {

	public static void main(String[] args) throws Exception {
                File file =  new File("../../rdfStore/str500.txt");

		List<String[]> statements = new ArrayList<String[]>();

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
		}
		br.close();
		int totalRes = 0;
		System.out.println("Elements "  + statements.size());
		int N_CYCLES = 1;

		List timeVec = new LinkedList();

		for (int i =0; i <N_CYCLES; i++)
		{
			List<Individual> stmt = new LinkedList<Individual>();
			double start = System.nanoTime();
			for (int k=0; k <statements.size(); k++)
			{
				stmt.add(Individual.create(statements.get(k)[0]));
                                stmt.add(Individual.create(statements.get(k)[1]));
                                stmt.add(Individual.create(statements.get(k)[2]));

				if (stmt.size() == 50000 * 3) {
		                        DataStore store = new DataStore(DataStore.StoreType.Sequential);

	                        	store.addTriples(stmt.toArray(new Individual[stmt.size()]));
					System.out.println("Number of tuples after import: " + store.getTriplesCount());

					Prefixes prefixes = Prefixes.DEFAULT_IMMUTABLE_INSTANCE;
					TupleIterator tupleIterator = store.compileQuery("SELECT DISTINCT * WHERE{ ?s ?p  <http://example.org/int/99> . <http://example.org/int/0> ?p ?o }", prefixes, new Parameters());

					System.out.println("EVALUATING");
					totalRes += evaluateAndPrintResults(prefixes, tupleIterator);
					// When no longer needed, the iterator should be disposed so that all related resources are released.
					tupleIterator.dispose();

					stmt.clear();
					store.dispose();
				}
			}

                        double time = System.nanoTime() - start;
                        double duration = new Double (time / (double) 1000000);
                        timeVec.add(duration);

		}
		System.out.println("Number of total results are " + totalRes);


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

	}

	public static int evaluateAndPrintResults(Prefixes prefixes, TupleIterator tupleIterator) throws JRDFoxException {
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
				Resource resource = tupleIterator.getResource(termIndex);
			}
			++numberOfRows;
		}
		System.out.println("---------------------------------------------------------------------------------------");
		System.out.println("  The number of rows returned: " + numberOfRows);
		System.out.println("=======================================================================================");
		System.out.println();

		return numberOfRows;
	}
}
