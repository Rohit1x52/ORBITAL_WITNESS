from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ThreatLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class SituationReport(BaseModel):
    visual_signatures: str = Field(description="Observable characteristics in satellite imagery")
    primary_threat: str = Field(description="Immediate danger to life and infrastructure")
    severity_level: ThreatLevel = Field(description="Classification of threat severity")
    critical_factors: List[str] = Field(description="Key factors requiring immediate attention")
    confidence_assessment: str = Field(description="Analyst confidence in classification")


class ResponsePlan(BaseModel):
    immediate_actions: List[str] = Field(description="0-24 hour priority actions")
    resource_requirements: List[str] = Field(description="Specific assets needed")
    mitigation_strategy: str = Field(description="Tactical containment approach")
    logistics_plan: str = Field(description="Supply chain and evacuation coordination")
    resilience_measures: List[str] = Field(description="Long-term prevention strategies")
    monitoring_enhancements: str = Field(description="Improved satellite monitoring protocols")
    context_reliability: str = Field(description="Assessment of knowledge base adequacy")


ENHANCED_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are ATLAS-7 (Advanced Tactical Land Analysis System), an elite AI-powered Geospatial Intelligence Analyst with specialized training in:
    
- Environmental Crisis Pattern Recognition
- Multi-spectral Satellite Imagery Interpretation
- Rapid Threat Assessment Protocols
- FEMA/UN Disaster Classification Standards

Your analysis must meet GEOINT Intelligence Community Directive (ICD) 203 standards for accuracy and brevity."""),
    
    ("human", """CLASSIFICATION ALERT: {label}
CONFIDENCE SCORE: {confidence}

Execute SITREP Generation Protocol:

SECTION ALPHA - Visual Indicators:
What spectral signatures, spatial patterns, and temporal changes define this event class in overhead imagery?

SECTION BRAVO - Threat Vector Analysis:
Identify the primary kinetic or environmental threat to civilian populations and critical infrastructure.

SECTION CHARLIE - Severity Assessment:
Quantify potential impact using standardized disaster metrics (casualties, displacement, economic damage).

SECTION DELTA - Time-Criticality:
Assess urgency window for intervention (Golden Hour, 72-hour window, extended response).

FORMATTING REQUIREMENTS:
- Maximum 75 words total
- Use GEOINT standard terminology
- Include quantifiable metrics where applicable
- Avoid speculation or hedging language
- Prioritize actionable intelligence

Begin SITREP:""")
])


ENHANCED_SUMMARY_PROMPT_STRUCTURED = ChatPromptTemplate.from_template(
    """You are an elite Geospatial Intelligence Analyst specializing in rapid crisis assessment using satellite reconnaissance.

EVENT CLASSIFICATION: {label}
DETECTION CONFIDENCE: {confidence}
ANALYSIS TIMESTAMP: {timestamp}

Generate a tactical Situation Report (SITREP) with the following intelligence components:

1. SPECTRAL SIGNATURES
What multi-spectral characteristics distinguish this event in satellite imagery? Consider:
- Visible spectrum anomalies (RGB patterns, color shifts)
- Thermal signatures (IR temperature deviations)
- Vegetation indices (NDVI changes for deforestation/drought)
- Water body analysis (NDWI for floods)
- Urban infrastructure changes (built-up area modifications)

2. THREAT VECTOR ANALYSIS
Primary danger classification:
- Human casualty potential (immediate/delayed)
- Infrastructure vulnerability (buildings, roads, utilities)
- Environmental contamination risk
- Economic impact zone radius
- Population displacement likelihood

3. SEVERITY QUANTIFICATION
Using standard disaster metrics:
- Expected casualty range
- Affected population estimate
- Infrastructure damage scale (1-5)
- Recovery timeline projection
- Resource mobilization urgency

4. TACTICAL RESPONSE WINDOW
- Golden Hour opportunities (0-1 hour)
- Critical intervention period (1-72 hours)
- Stabilization phase (3-14 days)
- Long-term recovery needs

FORMAT REQUIREMENTS:
- 100-150 words maximum
- Military precision language
- Include numerical estimates
- No speculative language
- Prioritize life-safety intelligence

SITREP Output:
"""
)


SOLUTION_PROMPT_ENHANCED = ChatPromptTemplate.from_messages([
    ("system", """You are COMMANDER-NEXUS, a Senior Crisis Operations AI system integrating:

- UN Office for the Coordination of Humanitarian Affairs (OCHA) protocols
- FEMA National Response Framework (NRF) guidelines
- NATO Crisis Response System Operations
- World Health Organization Emergency Response protocols
- International Federation of Red Cross disaster management standards

You operate with full authority to coordinate multi-agency disaster response operations."""),
    
    ("human", """CRISIS ACTIVATION CODE: ALPHA-{event_class}

INTELLIGENCE PACKAGE:
Threat Classification: {event_class}
Field Assessment: {summary}
Confidence Level: {confidence}

RETRIEVED OPERATIONAL PROTOCOLS:
{context}

MISSION DIRECTIVE:
Generate a comprehensive Strategic Response Action Plan (SRAP) using military-grade operational planning structure.

═══════════════════════════════════════════════════
PHASE 1: IMMEDIATE RESPONSE (H+0 to H+24)
═══════════════════════════════════════════════════

PRIORITY ACTIONS (Life-Safety Focus):
1. [Action 1 - Specify exact steps, responsible agencies, success metrics]
2. [Action 2 - Include coordination requirements and communication protocols]
3. [Action 3 - Define completion criteria and fallback procedures]

ASSET DEPLOYMENT MATRIX:
- Aerial Resources: [Drones, helicopters, fixed-wing - quantities and missions]
- Ground Assets: [Search/rescue teams, medical units, engineering corps - deployment locations]
- Support Infrastructure: [Field hospitals, staging areas, command posts - establishment timeline]
- Technology Systems: [Satellite monitoring, communication networks, data analytics]

COMMAND STRUCTURE:
- Incident Commander: [Role designation]
- Operations Chief: [Responsibilities]
- Logistics Coordinator: [Supply chain authority]

═══════════════════════════════════════════════════
PHASE 2: TACTICAL STABILIZATION (Days 1-30)
═══════════════════════════════════════════════════

CONTAINMENT STRATEGY (Event-Specific):
Based on {event_class} characteristics:
- Primary containment method: [Specific to threat type]
- Secondary stabilization measures: [Infrastructure protection]
- Environmental mitigation: [Pollution control, debris management]
- Perimeter security: [Access control, safety zones]

LOGISTICS OPERATIONS:
- Supply Chain: [Routes, distribution points, inventory management]
- Evacuation Protocols: [Primary/secondary routes, assembly points, transportation]
- Shelter Management: [Capacity planning, resource allocation, duration estimates]
- Medical Support: [Triage centers, hospital coordination, pharmaceuticals]
- Communications: [Emergency broadcast systems, inter-agency networks]

RECOVERY METRICS:
- Infrastructure restoration targets (%)
- Population return timeline
- Economic activity resumption goals
- Environmental remediation checkpoints

═══════════════════════════════════════════════════
PHASE 3: STRATEGIC RESILIENCE (30+ Days)
═══════════════════════════════════════════════════

POLICY FRAMEWORK:
- Regulatory updates: [Building codes, zoning laws, environmental protections]
- Governance improvements: [Emergency management authority, budget allocation]
- International cooperation: [Cross-border protocols, resource sharing agreements]

ENGINEERING SOLUTIONS:
- Infrastructure hardening: [Specific to {event_class} vulnerabilities]
- Early warning systems: [Sensor networks, prediction models, alert mechanisms]
- Redundancy planning: [Backup systems, alternative routes, failsafe designs]
- Green infrastructure: [Nature-based solutions, ecosystem restoration]

TECHNOLOGICAL INTEGRATION:
- Enhanced satellite monitoring: [Frequency, resolution, analysis algorithms]
- AI/ML prediction models: [Real-time risk assessment, pattern recognition]
- IoT sensor networks: [Ground truth validation, continuous monitoring]
- Data fusion platforms: [Multi-source integration, decision support systems]

COMMUNITY PREPAREDNESS:
- Training programs: [First responders, civilians, community leaders]
- Drill schedules: [Frequency, scope, evaluation criteria]
- Resource stockpiling: [Strategic reserves, distribution networks]
- Public awareness campaigns: [Education, communication channels]

═══════════════════════════════════════════════════
CRITICAL ASSESSMENT NOTES
═══════════════════════════════════════════════════

CONTEXT RELIABILITY: {context_quality}
If knowledge base is insufficient, default to:
- FEMA Emergency Support Functions (ESF) framework
- UN OCHA Cluster Approach protocols
- Sphere Humanitarian Standards
- National Incident Management System (NIMS) guidelines

RESOURCE CONSTRAINTS:
- Budget allocation estimates: [USD ranges]
- Personnel requirements: [Headcount by specialty]
- Timeline feasibility: [Realistic vs. optimal scenarios]
- Political/jurisdictional considerations: [Coordination challenges]

SUCCESS CRITERIA:
- Lives saved (target metrics)
- Infrastructure functionality (% restoration)
- Economic recovery indicators
- Environmental recovery benchmarks
- Community resilience scores

AUTHORIZATION LEVEL: EXECUTE IMMEDIATELY
All actions require simultaneous coordination with local, national, and international authorities.

Generate complete SRAP:""")
])


FEW_SHOT_EXAMPLES = [
    {
        "event_class": "wildfire",
        "summary": "Active fire front detected with rapid expansion rate. High thermal signature indicates extreme combustion temperatures. Vegetation density suggests sustained fuel availability.",
        "response": """PHASE 1 IMMEDIATE RESPONSE (H+0 to H+24):

PRIORITY ACTIONS:
1. Deploy aerial fire suppression fleet (12 fixed-wing tankers, 8 helicopters) to establish containment lines along southwestern perimeter within 4 hours
2. Execute mandatory evacuation of 15,000 residents in Zones A-D using primary Highway 101 and secondary Route 23, completion target 6 hours
3. Establish Incident Command Post at County Fairgrounds with liaison officers from CAL FIRE, FEMA, Red Cross operational within 90 minutes

ASSET DEPLOYMENT:
- Aerial: 20 aircraft (retardant drops, reconnaissance, medical evacuation)
- Ground: 450 firefighters, 85 engines, 12 bulldozers for firebreak construction
- Support: 3 field hospitals, 8 evacuation shelters (capacity 20,000)
- Technology: Drone thermal mapping every 30 minutes, satellite monitoring hourly

PHASE 2 TACTICAL STABILIZATION (Days 1-30):
Establish 3-mile firebreak using heavy equipment and controlled burns. Coordinate with meteorology teams for weather-based tactical adjustments. Deploy mop-up crews for hotspot elimination. Restore power grid in unaffected zones within 72 hours."""
    }
]


FEW_SHOT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["event_class", "summary"],
    template="""Event: {event_class}
Assessment: {summary}

Response Plan:
{response}"""
)


FEW_SHOT_SOLUTION_PROMPT = FewShotPromptTemplate(
    examples=FEW_SHOT_EXAMPLES,
    example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
    prefix="""You are a crisis response AI trained on thousands of disaster scenarios. Generate detailed, actionable response plans based on the following examples:""",
    suffix="""Now generate a response plan for:
Event: {event_class}
Assessment: {summary}
Context: {context}

Response Plan:""",
    input_variables=["event_class", "summary", "context"]
)


CONFIDENCE_WEIGHTED_PROMPT = ChatPromptTemplate.from_template(
    """You are analyzing a disaster event with CONFIDENCE LEVEL: {confidence}

CONFIDENCE INTERPRETATION:
- 0.90-1.00: High confidence - Full resource mobilization authorized
- 0.75-0.89: Good confidence - Standard response protocols
- 0.60-0.74: Moderate confidence - Cautious deployment with verification
- Below 0.60: Low confidence - Human analyst review required, preliminary staging only

EVENT CLASSIFICATION: {label}
DETECTED CONFIDENCE: {confidence}

Adjust your response plan based on confidence level:

HIGH CONFIDENCE (>0.75):
- Immediate full-scale response
- Pre-positioning of all assets
- Maximum resource allocation

MODERATE CONFIDENCE (0.60-0.74):
- Staged response with verification steps
- Conservative resource deployment
- Enhanced monitoring and re-assessment protocols
- Backup plans for misclassification scenarios

LOW CONFIDENCE (<0.60):
- Hold primary response pending verification
- Deploy reconnaissance assets only
- Request additional satellite passes
- Activate human expert review board
- Prepare multiple scenario response plans

INTELLIGENCE SUMMARY: {summary}
KNOWLEDGE BASE: {context}

Generate appropriate response plan scaled to confidence level:
"""
)


MULTI_THREAT_PROMPT = ChatPromptTemplate.from_template(
    """COMPLEX CRISIS SCENARIO DETECTED

Multiple threat classifications identified in the same geographic area:

PRIMARY THREAT: {primary_class} (Confidence: {primary_confidence})
SECONDARY THREAT: {secondary_class} (Confidence: {secondary_confidence})
TERTIARY FACTORS: {additional_factors}

CASCADING RISK ANALYSIS:
Assess potential compound effects:
- How does {primary_class} exacerbate {secondary_class}?
- What infrastructure failures could trigger additional hazards?
- Which populations face multiple concurrent threats?
- What resource conflicts may arise between response efforts?

INTEGRATED RESPONSE STRATEGY:
Generate a unified action plan that addresses all threat vectors simultaneously, prioritizing:
1. Life safety across all hazard types
2. Resource optimization for multi-threat scenarios
3. Prevention of cascade effects
4. Coordinated multi-agency response

KNOWLEDGE BASE: {context}

Develop comprehensive multi-hazard response plan:
"""
)


SUMMARY_PROMPT = ENHANCED_SUMMARY_PROMPT_STRUCTURED
SOLUTION_PROMPT = SOLUTION_PROMPT_ENHANCED


class PromptManager:
    def __init__(self):
        self.summary_prompt = ENHANCED_SUMMARY_PROMPT_STRUCTURED
        self.solution_prompt = SOLUTION_PROMPT_ENHANCED
        self.confidence_prompt = CONFIDENCE_WEIGHTED_PROMPT
        self.multi_threat_prompt = MULTI_THREAT_PROMPT
        self.few_shot_prompt = FEW_SHOT_SOLUTION_PROMPT
    
    def get_summary_prompt(
        self, 
        use_structured: bool = True
    ) -> ChatPromptTemplate:
        return (ENHANCED_SUMMARY_PROMPT_STRUCTURED 
                if use_structured 
                else ENHANCED_SUMMARY_PROMPT)
    
    def get_solution_prompt(
        self, 
        confidence: float,
        use_few_shot: bool = False,
        multi_threat: bool = False
    ) -> ChatPromptTemplate:
        if multi_threat:
            return self.multi_threat_prompt
        
        if confidence < 0.60:
            return self.confidence_prompt
        
        if use_few_shot and confidence > 0.75:
            return self.few_shot_prompt
        
        return self.solution_prompt
    
    def get_custom_prompt(
        self,
        template: str,
        input_variables: List[str]
    ) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(template)


if __name__ == "__main__":
    manager = PromptManager()
    
    test_inputs = {
        "label": "wildfire",
        "confidence": "0.87",
        "timestamp": "2024-01-24T10:30:00Z",
        "event_class": "wildfire",
        "summary": "Active fire detected with rapid expansion",
        "context": "Standard wildfire response protocols available",
        "context_quality": "High - Complete knowledge base available"
    }
    
    summary_prompt = manager.get_summary_prompt()
    formatted_summary = summary_prompt.format(**test_inputs)
    
    print("SUMMARY PROMPT:")
    print(formatted_summary)
    print("\n" + "="*80 + "\n")
    
    solution_prompt = manager.get_solution_prompt(confidence=0.87)
    formatted_solution = solution_prompt.format(**test_inputs)
    
    print("SOLUTION PROMPT:")
    print(formatted_solution)