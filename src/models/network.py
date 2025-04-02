"""
Market network analysis module for Yemen Market Integration analysis.

This module provides functionality for analyzing market integration patterns
through network analysis techniques, identifying central markets and
community structures within the broader market system.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import networkx as nx
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from yemen_market_integration.utils.error_handler import handle_errors
from yemen_market_integration.utils.decorators import timer
from yemen_market_integration.utils.m3_utils import m3_optimized, tiered_cache, optimize_array_computation
from yemen_market_integration.utils.validation import validate_dataframe, validate_network, raise_if_invalid
from yemen_market_integration.utils.multiple_testing import apply_multiple_testing_correction

# Initialize module logger
logger = logging.getLogger(__name__)


@dataclass
class MarketCentrality:
    """Container for market centrality metrics."""
    market: str
    degree: float = 0.0
    betweenness: float = 0.0
    closeness: float = 0.0
    eigenvector: float = 0.0
    pagerank: float = 0.0
    strength: float = 0.0  # Weighted degree centrality
    
    def as_dict(self) -> Dict[str, float]:
        """Convert centrality metrics to dictionary."""
        return {
            'market': self.market,
            'degree': self.degree,
            'betweenness': self.betweenness,
            'closeness': self.closeness,
            'eigenvector': self.eigenvector,
            'pagerank': self.pagerank,
            'strength': self.strength
        }


@dataclass
class NetworkCommunity:
    """Container for a network community of markets."""
    community_id: int
    markets: List[str] = field(default_factory=list)
    size: int = 0
    centrality: float = 0.0  # Average centrality of markets in community
    internal_density: float = 0.0  # Density of connections within community
    external_connections: int = 0  # Number of connections to other communities
    
    def add_market(self, market: str) -> None:
        """Add a market to the community."""
        if market not in self.markets:
            self.markets.append(market)
            self.size = len(self.markets)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert community data to dictionary."""
        return {
            'community_id': self.community_id,
            'markets': self.markets,
            'size': self.size,
            'centrality': self.centrality,
            'internal_density': self.internal_density,
            'external_connections': self.external_connections
        }


class MarketNetworkAnalysis:
    """
    Network-based analysis of market integration patterns.
    
    This class treats markets as nodes in a network with price transmission
    strength as edge weights. It provides functionality for centrality measures,
    community detection, and other network analytics to identify market 
    integration patterns and key markets within the system.
    """
    
    @timer
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def __init__(
        self, 
        market_results: Dict[str, Dict[str, Any]],
        integration_measure: str = 'half_life',
        weight_transform: str = 'inverse',
        weight_threshold: Optional[float] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize market network analysis.
        
        Parameters
        ----------
        market_results : Dict[str, Dict[str, Any]]
            Results from market pair analysis, with keys in format "market1_market2"
        integration_measure : str, default='half_life'
            Measure to use for edge weights:
            - 'half_life': Half-life of adjustment from threshold model
            - 'cointegration_stat': Cointegration test statistic
            - 'threshold': Threshold value
            - 'adjustment_speed': Speed of adjustment coefficient
            - 'p_value': P-value from cointegration test (reversed)
        weight_transform : str, default='inverse'
            Transformation to apply to the integration measure:
            - 'inverse': 1/(measure+1) (smaller half-life = stronger integration)
            - 'negative': -measure (more negative = stronger integration)
            - 'direct': measure (as-is)
        weight_threshold : float, optional
            Minimum weight threshold for including an edge
        max_workers : int, optional
            Maximum number of workers for parallel processing
        """
        self.market_results = market_results
        self.integration_measure = integration_measure
        self.weight_transform = weight_transform
        self.weight_threshold = weight_threshold
        
        # Set max workers
        self.max_workers = max_workers
        if max_workers is None:
            import multiprocessing as mp
            self.max_workers = max(1, mp.cpu_count() - 1)
        
        # Initialize network components
        self.graph = None
        self.centrality_measures = {}
        self.communities = None
        self.community_objects = []
        self.modularity = 0.0
        
        # Build network from market results
        self._build_network()
        
        # Log network properties
        logger.info(
            f"Built market network with {self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges"
        )
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError), reraise=True)
    def _get_weight_from_result(
        self, 
        result: Dict[str, Any]
    ) -> Optional[float]:
        """
        Extract and transform weight from a market pair result.
        
        Parameters
        ----------
        result : Dict[str, Any]
            Analysis result for a market pair
            
        Returns
        -------
        float or None
            Transformed weight value, or None if not available
        """
        # Default value if measure not found
        measure_value = None
        
        # Extract based on specified measure
        if self.integration_measure == 'half_life':
            if 'threshold' in result and 'half_life' in result.get('threshold', {}):
                measure_value = result['threshold']['half_life']
        
        elif self.integration_measure == 'cointegration_stat':
            if 'cointegration' in result and 'statistic' in result.get('cointegration', {}):
                measure_value = result['cointegration']['statistic']
        
        elif self.integration_measure == 'threshold':
            if 'threshold' in result and 'threshold_value' in result.get('threshold', {}):
                measure_value = result['threshold']['threshold_value']
        
        elif self.integration_measure == 'adjustment_speed':
            if 'threshold' in result and 'adjustment_speed' in result.get('threshold', {}):
                measure_value = result['threshold']['adjustment_speed']
        
        elif self.integration_measure == 'p_value':
            if 'cointegration' in result and 'p_value' in result.get('cointegration', {}):
                measure_value = result['cointegration']['p_value']
        
        else:
            logger.warning(f"Unknown integration measure: {self.integration_measure}")
            return None
        
        # Return None if measure not found
        if measure_value is None:
            return None
        
        # Apply transformation
        if self.weight_transform == 'inverse':
            # Add 1 to avoid division by zero, invert so smaller values (like half-life) give higher weights
            return 1.0 / (float(measure_value) + 1.0)
        
        elif self.weight_transform == 'negative':
            # Negative transformation for measures where more negative is stronger
            return -float(measure_value)
        
        elif self.weight_transform == 'direct':
            # Use as-is
            return float(measure_value)
        
        else:
            logger.warning(f"Unknown weight transformation: {self.weight_transform}")
            return float(measure_value)
    
    @m3_optimized
    def _build_network(self) -> None:
        """
        Build network from market pair results.
        
        Constructs a weighted graph where nodes are markets and edge weights
        represent integration strength between markets.
        """
        # Create empty graph
        G = nx.Graph()
        
        # Add edges from market pairs
        for pair_key, result in self.market_results.items():
            try:
                # Extract markets from pair key
                markets = pair_key.split('_')
                if len(markets) != 2:
                    logger.warning(f"Invalid market pair key: {pair_key}")
                    continue
                
                market1, market2 = markets
                
                # Only add edge if markets are cointegrated
                is_cointegrated = result.get('cointegration', {}).get('cointegrated', False)
                
                if is_cointegrated:
                    # Calculate integration strength
                    weight = self._get_weight_from_result(result)
                    
                    # Skip if weight is None or below threshold
                    if weight is None:
                        continue
                    
                    if self.weight_threshold is not None and weight < self.weight_threshold:
                        continue
                    
                    # Add edge with weight
                    G.add_edge(market1, market2, weight=weight)
                
            except Exception as e:
                logger.warning(f"Error adding edge for pair {pair_key}: {e}")
        
        # Set graph
        self.graph = G
        
        # Add any isolated markets that exist in the results but not in any pairs
        all_markets = set()
        for pair_key in self.market_results.keys():
            markets = pair_key.split('_')
            if len(markets) == 2:
                all_markets.update(markets)
        
        # Add any missing markets as isolated nodes
        for market in all_markets:
            if market not in G:
                G.add_node(market)
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def calculate_centrality(self) -> Dict[str, MarketCentrality]:
        """
        Calculate centrality measures for markets.
        
        Computes various centrality metrics to identify key markets in the network.
        
        Returns
        -------
        Dict[str, MarketCentrality]
            Dictionary of centrality measures for each market
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.warning("Network not built or empty. Cannot calculate centrality.")
            return {}
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        
        # For larger networks, use parallel algorithms
        if self.graph.number_of_nodes() > 100:
            try:
                import multiprocessing as mp
                from joblib import Parallel, delayed
                
                # Calculate centrality measures in parallel
                with Parallel(n_jobs=self.max_workers) as parallel:
                    betweenness_centrality = nx.betweenness_centrality(
                        self.graph, weight='weight', k=min(50, self.graph.number_of_nodes())
                    )
                    closeness_centrality = parallel(
                        delayed(nx.closeness_centrality)(self.graph, u, distance='weight')
                        for u in self.graph.nodes()
                    )
                    # Convert list of closeness values to dictionary
                    closeness_centrality = dict(zip(self.graph.nodes(), closeness_centrality))
            except (ImportError, Exception) as e:
                logger.warning(f"Error in parallel centrality calculation: {e}. Using sequential methods.")
                betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight')
                closeness_centrality = nx.closeness_centrality(self.graph, distance='weight')
        else:
            betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight')
            closeness_centrality = nx.closeness_centrality(self.graph, distance='weight')
        
        # These are always calculated sequentially
        try:
            # Use power iteration method for large graphs
            if self.graph.number_of_nodes() > 500:
                eigenvector_centrality = nx.eigenvector_centrality_numpy(self.graph, weight='weight')
            else:
                eigenvector_centrality = nx.eigenvector_centrality(
                    self.graph, max_iter=1000, tol=1e-6, weight='weight'
                )
        except (Exception, nx.PowerIterationFailedConvergence) as e:
            logger.warning(f"Eigenvector centrality calculation failed: {e}. Using degree as fallback.")
            eigenvector_centrality = degree_centrality
            
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Calculate strength (weighted degree) centrality
        strength = {}
        for node in self.graph.nodes():
            strength[node] = sum(
                self.graph[node][neighbor].get('weight', 1.0)
                for neighbor in self.graph.neighbors(node)
            )
        # Normalize strength
        if len(strength) > 0:
            max_strength = max(strength.values()) if strength else 1.0
            strength = {k: v / max_strength for k, v in strength.items()}
        
        # Organize results by market
        centrality = {}
        for market in self.graph.nodes():
            centrality[market] = MarketCentrality(
                market=market,
                degree=degree_centrality.get(market, 0),
                betweenness=betweenness_centrality.get(market, 0),
                closeness=closeness_centrality.get(market, 0),
                eigenvector=eigenvector_centrality.get(market, 0),
                pagerank=pagerank.get(market, 0),
                strength=strength.get(market, 0)
            )
        
        self.centrality_measures = centrality
        
        # Log top markets by various centrality measures
        top_markets = self._get_top_markets(n=5)
        logger.info(f"Top markets by centrality: {top_markets}")
        
        return centrality
    
    def _get_top_markets(self, n: int = 5) -> Dict[str, List[str]]:
        """
        Get top n markets by each centrality measure.
        
        Parameters
        ----------
        n : int
            Number of top markets to return
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with centrality types as keys and lists of top markets as values
        """
        if not self.centrality_measures:
            return {}
        
        top_markets = {}
        
        # Get top markets by each measure
        measure_dict = {
            'degree': [(m.market, m.degree) for m in self.centrality_measures.values()],
            'betweenness': [(m.market, m.betweenness) for m in self.centrality_measures.values()],
            'closeness': [(m.market, m.closeness) for m in self.centrality_measures.values()],
            'eigenvector': [(m.market, m.eigenvector) for m in self.centrality_measures.values()],
            'pagerank': [(m.market, m.pagerank) for m in self.centrality_measures.values()],
            'strength': [(m.market, m.strength) for m in self.centrality_measures.values()]
        }
        
        for measure, values in measure_dict.items():
            # Sort by value (descending) and get top n
            sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
            top_markets[measure] = [market for market, _ in sorted_values[:n]]
        
        return top_markets
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def detect_communities(
        self,
        method: str = 'louvain',
        resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect market integration communities.
        
        Uses community detection algorithms to identify
        clusters of highly integrated markets.
        
        Parameters
        ----------
        method : str
            Community detection method:
            - 'louvain': Louvain method (modularity optimization)
            - 'leiden': Leiden method (improved Louvain)
            - 'label_propagation': Label propagation algorithm
            - 'fluid': Fluid communities algorithm
        resolution : float
            Resolution parameter for Louvain/Leiden methods.
            Higher values produce more communities.
            
        Returns
        -------
        Dict[str, int]
            Dictionary mapping markets to community IDs
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.warning("Network not built or empty. Cannot detect communities.")
            return {}
        
        # Detect communities
        if method == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(self.graph, resolution=resolution, weight='weight')
                self.modularity = community_louvain.modularity(communities, self.graph, weight='weight')
            except ImportError:
                logger.warning("python-louvain package not found. Using label propagation instead.")
                method = 'label_propagation'
        
        if method == 'leiden':
            try:
                import leidenalg
                import igraph as ig
                
                # Convert networkx graph to igraph
                edges = list(self.graph.edges(data='weight'))
                g = ig.Graph()
                g.add_vertices(list(self.graph.nodes()))
                g.add_edges([(u, v) for u, v, _ in edges])
                g.es['weight'] = [w if w is not None else 1.0 for _, _, w in edges]
                
                partition = leidenalg.find_partition(
                    g, 
                    leidenalg.ModularityVertexPartition, 
                    weights='weight',
                    resolution_parameter=resolution
                )
                
                # Convert to dict format
                communities = {}
                for i, community in enumerate(partition):
                    for vertex in community:
                        node_name = g.vs[vertex]['name']
                        communities[node_name] = i
                
                # Calculate modularity
                self.modularity = partition.quality()
                
            except ImportError:
                logger.warning("leidenalg or igraph package not found. Using label propagation instead.")
                method = 'label_propagation'
        
        if method == 'label_propagation':
            # Label propagation
            communities = {node: i for i, comm in enumerate(nx.algorithms.community.label_propagation_communities(self.graph)) 
                           for node in comm}
            
            # Calculate modularity
            from networkx.algorithms.community import modularity
            partition = {}
            for node, comm_id in communities.items():
                if comm_id not in partition:
                    partition[comm_id] = []
                partition[comm_id].append(node)
            
            self.modularity = modularity(self.graph, partition.values(), weight='weight')
            
        elif method == 'fluid':
            try:
                # Determine k (number of communities) heuristically based on network size
                k = max(2, int(np.sqrt(self.graph.number_of_nodes()) / 2))
                
                fluid = nx.algorithms.community.asyn_fluidc(self.graph, k, weight='weight')
                communities = {node: i for i, comm in enumerate(fluid) for node in comm}
                
                # Calculate modularity
                from networkx.algorithms.community import modularity
                partition = {}
                for node, comm_id in communities.items():
                    if comm_id not in partition:
                        partition[comm_id] = []
                    partition[comm_id].append(node)
                
                self.modularity = modularity(self.graph, partition.values(), weight='weight')
                
            except Exception as e:
                logger.warning(f"Fluid communities algorithm failed: {e}. Using label propagation instead.")
                communities = {node: i for i, comm in enumerate(nx.algorithms.community.label_propagation_communities(self.graph)) 
                               for node in comm}
        
        self.communities = communities
        
        # Analyze and create community objects
        self._analyze_communities()
        
        # Count markets in each community
        community_counts = {}
        for community_id in set(communities.values()):
            count = sum(1 for c in communities.values() if c == community_id)
            community_counts[community_id] = count
        
        logger.info(
            f"Detected {len(community_counts)} market communities with modularity {self.modularity:.4f}"
        )
        for comm_id, count in sorted(community_counts.items()):
            logger.info(f"Community {comm_id}: {count} markets")
        
        return communities
    
    def _analyze_communities(self) -> None:
        """
        Analyze detected communities in detail.
        
        This method calculates community-level metrics and creates community objects.
        """
        if self.communities is None:
            logger.warning("Communities not detected. Call detect_communities() first.")
            return
        
        # Clear previous community objects
        self.community_objects = []
        
        # Group markets by community
        communities_dict = {}
        for market, comm_id in self.communities.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(market)
        
        # Create community objects
        for comm_id, markets in communities_dict.items():
            comm_obj = NetworkCommunity(
                community_id=comm_id,
                markets=markets,
                size=len(markets)
            )
            
            # Calculate internal density
            if len(markets) > 1:
                internal_edges = 0
                possible_edges = (len(markets) * (len(markets) - 1)) / 2
                
                for i, m1 in enumerate(markets):
                    for m2 in markets[i+1:]:
                        if self.graph.has_edge(m1, m2):
                            internal_edges += 1
                
                comm_obj.internal_density = internal_edges / possible_edges if possible_edges > 0 else 0
            
            # Calculate external connections
            external_connections = 0
            for market in markets:
                for neighbor in self.graph.neighbors(market):
                    if self.communities.get(neighbor) != comm_id:
                        external_connections += 1
            
            comm_obj.external_connections = external_connections
            
            # Calculate average centrality if centrality measures exist
            if self.centrality_measures:
                avg_centrality = {
                    'degree': 0.0,
                    'betweenness': 0.0,
                    'eigenvector': 0.0
                }
                
                for market in markets:
                    if market in self.centrality_measures:
                        avg_centrality['degree'] += self.centrality_measures[market].degree
                        avg_centrality['betweenness'] += self.centrality_measures[market].betweenness
                        avg_centrality['eigenvector'] += self.centrality_measures[market].eigenvector
                
                # Average the centrality values
                for measure in avg_centrality:
                    avg_centrality[measure] /= len(markets) if len(markets) > 0 else 1
                
                # Use eigenvector centrality as the community centrality
                comm_obj.centrality = avg_centrality['eigenvector']
            
            # Add to community objects list
            self.community_objects.append(comm_obj)
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def analyze_inter_community_integration(self) -> Dict[str, Any]:
        """
        Analyze integration patterns between market communities.
        
        Measures the strength and patterns of integration between different
        market communities to identify structural patterns and barriers
        to integration.
        
        Returns
        -------
        Dict[str, Any]
            Cross-community integration metrics and patterns
        """
        if self.graph is None or self.communities is None:
            logger.warning("Network not built or communities not detected yet.")
            return {}
        
        # Group nodes by community
        community_nodes = {}
        for node, community in self.communities.items():
            if community not in community_nodes:
                community_nodes[community] = []
            community_nodes[community].append(node)
        
        # Calculate inter-community edges and weights
        inter_community_edges = {}
        
        for comm1, comm2 in [(i, j) for i in community_nodes for j in community_nodes if i < j]:
            edges = []
            weights = []
            
            for u in community_nodes[comm1]:
                for v in community_nodes[comm2]:
                    if self.graph.has_edge(u, v):
                        edges.append((u, v))
                        weights.append(self.graph[u][v].get('weight', 1.0))
            
            if edges:
                inter_community_edges[f"{comm1}_{comm2}"] = {
                    'count': len(edges),
                    'avg_weight': np.mean(weights) if weights else 0,
                    'edges': edges,
                    'weights': weights,
                    'community1_size': len(community_nodes[comm1]),
                    'community2_size': len(community_nodes[comm2]),
                    'connection_density': len(edges) / (len(community_nodes[comm1]) * len(community_nodes[comm2]))
                }
        
        # Calculate overall inter vs intra community metrics
        intra_community_edges = 0
        intra_community_weights = []
        inter_community_count = 0
        inter_community_weights = []
        
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            u_comm = self.communities.get(u)
            v_comm = self.communities.get(v)
            
            if u_comm == v_comm:
                intra_community_edges += 1
                intra_community_weights.append(weight)
            else:
                inter_community_count += 1
                inter_community_weights.append(weight)
        
        # Calculate E-I index (External-Internal index)
        total_edges = intra_community_edges + inter_community_count
        ei_index = (inter_community_count - intra_community_edges) / total_edges if total_edges > 0 else 0
        
        # Check if communities are assortative (more internal than external connections)
        is_assortative = intra_community_edges > inter_community_count
        
        # Calculate average weights
        avg_intra_weight = np.mean(intra_community_weights) if intra_community_weights else 0
        avg_inter_weight = np.mean(inter_community_weights) if inter_community_weights else 0
        
        # Compile results
        result = {
            'inter_community_connections': inter_community_edges,
            'intra_community_edges': intra_community_edges,
            'inter_community_edges': inter_community_count,
            'total_edges': total_edges,
            'modularity': self.modularity,
            'ei_index': ei_index,
            'is_assortative': is_assortative,
            'avg_intra_community_weight': avg_intra_weight,
            'avg_inter_community_weight': avg_inter_weight,
            'weight_ratio': avg_intra_weight / avg_inter_weight if avg_inter_weight > 0 else float('inf')
        }
        
        logger.info(
            f"Community structure analysis: "
            f"Modularity={self.modularity:.4f}, E-I Index={ei_index:.4f}, "
            f"{'Assortative' if is_assortative else 'Non-assortative'} mixing"
        )
        
        return result
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def identify_key_connector_markets(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify markets that play key roles in connecting different communities.
        
        Parameters
        ----------
        n : int, default=5
            Number of top connector markets to return
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (market, score) tuples for top connector markets
        """
        if self.graph is None or self.communities is None:
            logger.warning("Network not built or communities not detected yet.")
            return []
        
        # Calculate cross-community connections for each market
        connector_scores = {}
        
        for node in self.graph.nodes():
            node_community = self.communities.get(node)
            cross_connections = 0
            total_connections = 0
            
            for neighbor in self.graph.neighbors(node):
                total_connections += 1
                if self.communities.get(neighbor) != node_community:
                    cross_connections += 1
            
            # Market has connections
            if total_connections > 0:
                # Score is proportion of connections that are cross-community
                connector_scores[node] = cross_connections / total_connections
            else:
                connector_scores[node] = 0.0
        
        # Sort by score (descending) and return top n
        top_connectors = sorted(connector_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        logger.info(f"Top connector markets: {top_connectors}")
        
        return top_connectors
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def identify_spatial_barriers(self) -> Dict[str, Any]:
        """
        Identify potential spatial barriers to market integration.
        
        This method analyzes if community structure aligns with
        geographic barriers like conflict zones or transportation challenges.
        
        Returns
        -------
        Dict[str, Any]
            Information about potential spatial barriers to market integration
        """
        if self.graph is None or self.communities is None:
            logger.warning("Network not built or communities not detected yet.")
            return {}
        
        # This is a placeholder for when geographic data is available
        # Actual implementation would require market location data and conflict/geography information
        
        # For now, just return the community structure information
        return {
            'communities': {comm_id: [node for node, c in self.communities.items() if c == comm_id]
                           for comm_id in set(self.communities.values())},
            'modularity': self.modularity,
            'note': "Full spatial barrier analysis requires geographic data integration"
        }
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def compute_network_resilience(self) -> Dict[str, Any]:
        """
        Analyze network resilience to market disruptions.
        
        Evaluates how the removal of key markets or connections affects
        the overall integration of the market system.
        
        Returns
        -------
        Dict[str, Any]
            Network resilience metrics
        """
        if self.graph is None:
            logger.warning("Network not built. Cannot analyze resilience.")
            return {}
        
        # Make a copy of the graph to avoid modifying the original
        G = self.graph.copy()
        
        # Calculate baseline network metrics
        baseline_metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'connectivity': nx.average_node_connectivity(G) if G.number_of_edges() > 0 else 0,
            'avg_shortest_path': nx.average_shortest_path_length(G, weight='weight') 
                                if nx.is_connected(G) and G.number_of_edges() > 0 else float('inf'),
            'clustering': nx.average_clustering(G, weight='weight')
        }
        
        # Analyze resilience by removing top centrality nodes
        if not self.centrality_measures:
            self.calculate_centrality()
        
        # Get top 5 markets by different centrality measures
        top_markets = {}
        for measure in ['degree', 'betweenness', 'eigenvector']:
            top_markets[measure] = sorted(
                [(m.market, getattr(m, measure)) for m in self.centrality_measures.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        
        # Calculate impact of removing each set of top markets
        resilience_results = {}
        
        for measure, markets in top_markets.items():
            impact = []
            
            # Create a copy of the graph
            G_copy = G.copy()
            
            # Remove markets one by one and measure impact
            for i, (market, _) in enumerate(markets):
                if market in G_copy:
                    G_copy.remove_node(market)
                    
                    # Calculate new metrics
                    new_metrics = {}
                    
                    # Handle disconnected graphs
                    if nx.is_connected(G_copy) and G_copy.number_of_edges() > 0:
                        new_metrics = {
                            'n_nodes': G_copy.number_of_nodes(),
                            'n_edges': G_copy.number_of_edges(),
                            'connectivity': nx.average_node_connectivity(G_copy),
                            'avg_shortest_path': nx.average_shortest_path_length(G_copy, weight='weight'),
                            'clustering': nx.average_clustering(G_copy, weight='weight')
                        }
                    else:
                        # For disconnected graphs, use largest connected component
                        largest_cc = max(nx.connected_components(G_copy), key=len)
                        largest_subgraph = G_copy.subgraph(largest_cc).copy()
                        
                        if largest_subgraph.number_of_edges() > 0:
                            new_metrics = {
                                'n_nodes': G_copy.number_of_nodes(),
                                'n_edges': G_copy.number_of_edges(),
                                'connectivity': nx.average_node_connectivity(largest_subgraph),
                                'avg_shortest_path': nx.average_shortest_path_length(largest_subgraph, weight='weight'),
                                'clustering': nx.average_clustering(G_copy, weight='weight'),
                                'largest_cc_size': len(largest_cc),
                                'n_components': nx.number_connected_components(G_copy)
                            }
                        else:
                            new_metrics = {
                                'n_nodes': G_copy.number_of_nodes(),
                                'n_edges': G_copy.number_of_edges(),
                                'connectivity': 0,
                                'avg_shortest_path': float('inf'),
                                'clustering': 0,
                                'largest_cc_size': len(largest_cc),
                                'n_components': nx.number_connected_components(G_copy)
                            }
                    
                    # Calculate relative changes from baseline
                    relative_changes = {}
                    for metric, val in baseline_metrics.items():
                        if metric in new_metrics:
                            if val != 0 and val != float('inf'):
                                relative_changes[metric] = (new_metrics[metric] - val) / val
                            else:
                                relative_changes[metric] = float('inf') if new_metrics[metric] > 0 else 0
                    
                    impact.append({
                        'removed_markets': [m[0] for m in markets[:i+1]],
                        'metrics': new_metrics,
                        'relative_changes': relative_changes
                    })
            
            resilience_results[measure] = impact
        
        # Overall resilience score (higher is better)
        avg_impact = np.mean([
            abs(impact[-1]['relative_changes'].get('connectivity', 0)) 
            for impact in resilience_results.values()
        ])
        
        resilience_score = 1 / (1 + avg_impact) if avg_impact > 0 else 1.0
        
        result = {
            'baseline_metrics': baseline_metrics,
            'resilience_by_centrality': resilience_results,
            'resilience_score': resilience_score
        }
        
        logger.info(f"Network resilience analysis complete. Resilience score: {resilience_score:.4f}")
        
        return result
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def visualize_network(
        self,
        output_path: Optional[str] = None,
        show_communities: bool = True,
        show_weights: bool = True,
        node_size_by: str = 'degree',
        layout: str = 'spring'
    ) -> Optional[plt.Figure]:
        """
        Visualize the market integration network.
        
        Creates a network visualization with nodes as markets and edges
        representing integration relationships.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the visualization. If None, the plot is returned.
        show_communities : bool, default=True
            Whether to color nodes by community
        show_weights : bool, default=True
            Whether to vary edge width by weight
        node_size_by : str, default='degree'
            Centrality measure to use for node size:
            - 'degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'strength'
        layout : str, default='spring'
            Network layout algorithm:
            - 'spring': Force-directed layout
            - 'circular': Circular layout
            - 'kamada_kawai': Kamada-Kawai layout
            - 'spectral': Spectral layout
            
        Returns
        -------
        matplotlib.figure.Figure or None
            Figure object if output_path is None, otherwise None
        """
        if self.graph is None:
            logger.warning("Network not built. Cannot visualize.")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get node positions based on layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, weight='weight', seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(self.graph, weight='weight')
            except:
                logger.warning("Kamada-Kawai layout failed. Using spring layout.")
                pos = nx.spring_layout(self.graph, weight='weight', seed=42)
        elif layout == 'spectral':
            try:
                pos = nx.spectral_layout(self.graph)
            except:
                logger.warning("Spectral layout failed. Using spring layout.")
                pos = nx.spring_layout(self.graph, weight='weight', seed=42)
        else:
            logger.warning(f"Unknown layout: {layout}. Using spring layout.")
            pos = nx.spring_layout(self.graph, weight='weight', seed=42)
        
        # Get node sizes based on centrality
        if not self.centrality_measures:
            self.calculate_centrality()
        
        node_sizes = []
        for node in self.graph.nodes():
            if node in self.centrality_measures:
                if node_size_by == 'degree':
                    size = 300 * (self.centrality_measures[node].degree + 0.1)
                elif node_size_by == 'betweenness':
                    size = 300 * (self.centrality_measures[node].betweenness + 0.1)
                elif node_size_by == 'closeness':
                    size = 300 * (self.centrality_measures[node].closeness + 0.1)
                elif node_size_by == 'eigenvector':
                    size = 300 * (self.centrality_measures[node].eigenvector + 0.1)
                elif node_size_by == 'pagerank':
                    size = 300 * (self.centrality_measures[node].pagerank + 0.1)
                elif node_size_by == 'strength':
                    size = 300 * (self.centrality_measures[node].strength + 0.1)
                else:
                    size = 100
            else:
                size = 100
            node_sizes.append(size)
        
        # Get edge widths based on weights
        edge_widths = []
        if show_weights:
            for u, v, data in self.graph.edges(data=True):
                width = data.get('weight', 1.0)
                # Scale width for visibility
                width = 1 + 3 * width
                edge_widths.append(width)
        else:
            edge_widths = [1] * self.graph.number_of_edges()
        
        # Color nodes by community
        node_colors = []
        if show_communities and self.communities:
            for node in self.graph.nodes():
                comm_id = self.communities.get(node, 0)
                node_colors.append(comm_id)
        else:
            node_colors = ['skyblue'] * self.graph.number_of_nodes()
        
        # Draw the network
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_size=node_sizes,
            node_color=node_colors if show_communities else 'skyblue',
            cmap=plt.cm.tab20 if show_communities else None,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=edge_widths,
            edge_color='gray',
            alpha=0.6
        )
        
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=10,
            font_family='sans-serif'
        )
        
        plt.title('Market Integration Network', fontsize=15)
        plt.axis('off')
        
        # Add legend for communities
        if show_communities and self.communities:
            comm_ids = sorted(set(self.communities.values()))
            cmap = plt.cm.tab20
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=cmap(i % 20), markersize=10,
                          label=f'Community {i}')
                for i in comm_ids
            ]
            plt.legend(handles=legend_elements, loc='best')
        
        # Save or show the figure
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Network visualization saved to {output_path}")
            return None
        else:
            plt.tight_layout()
            return plt.gcf()
