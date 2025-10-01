test_cases = [
    # Health
    {
        "claim1": "Regular physical exercise significantly reduces the risk of heart disease.",
        "claim2": "Studies show that individuals who exercise 30 minutes daily have a 35% lower risk of cardiovascular issues.",
        "expected": "Support"
    },
    {
        "claim1": "Vegan diets provide all necessary nutrients for human health.",
        "claim2": "Vegan diets often lack essential nutrients like vitamin B12 and iron, requiring supplements.",
        "expected": "Attack"
    },
    {
        "claim1": "Sleeping less than 7 hours per night impairs cognitive function.",
        "claim2": "Chronic sleep deprivation is linked to memory problems and reduced concentration.",
        "expected": "Support"
    },
    {
        "claim1": "Meditation improves mental health and reduces stress.",
        "claim2": "Clinical studies show that regular meditation lowers cortisol levels and anxiety.",
        "expected": "Support"
    },
    {
        "claim1": "Drinking coffee daily is harmful to health.",
        "claim2": "Moderate coffee consumption is linked to reduced risk of Parkinson’s and type 2 diabetes.",
        "expected": "Attack"
    },

    # Technology
    {
        "claim1": "Artificial intelligence will replace most human jobs within 20 years.",
        "claim2": "AI is creating new job categories that require human oversight and creativity.",
        "expected": "Attack"
    },
    {
        "claim1": "5G technology poses serious health risks due to radiation.",
        "claim2": "Scientific studies have found no evidence linking 5G radiation to health problems.",
        "expected": "Attack"
    },
    {
        "claim1": "Blockchain technology ensures transparent and secure transactions.",
        "claim2": "Blockchain's decentralized nature makes it resistant to fraud and censorship.",
        "expected": "Support"
    },
    {
        "claim1": "Self-driving cars will drastically reduce traffic accidents.",
        "claim2": "Autonomous vehicles eliminate human error, the leading cause of accidents.",
        "expected": "Support"
    },
    {
        "claim1": "Virtual reality will replace traditional forms of entertainment.",
        "claim2": "High equipment costs and motion sickness limit VR adoption.",
        "expected": "Attack"
    },

    # Environnement
    {
        "claim1": "Nuclear energy is the cleanest solution to climate change.",
        "claim2": "Nuclear power plants produce zero carbon emissions during operation.",
        "expected": "Support"
    },
    {
        "claim1": "Plastic recycling effectively reduces ocean pollution.",
        "claim2": "Only 9% of all plastic waste ever produced has been recycled.",
        "expected": "Attack"
    },
    {
        "claim1": "Electric cars are worse for the environment than gasoline cars.",
        "claim2": "Over their lifetime, electric vehicles produce 50% less CO2 than gasoline cars.",
        "expected": "Attack"
    },
    {
        "claim1": "Planting trees is the most effective solution to climate change.",
        "claim2": "Trees absorb CO2, but large-scale reduction of fossil fuel use is still necessary.",
        "expected": "Attack"
    },
    {
        "claim1": "Meat production is a major driver of climate change.",
        "claim2": "Livestock farming accounts for 14.5% of global greenhouse gas emissions.",
        "expected": "Support"
    },

    # Education
    {
        "claim1": "Standardized testing accurately measures student knowledge.",
        "claim2": "Standardized tests often reflect socioeconomic status rather than true ability.",
        "expected": "Attack"
    },
    {
        "claim1": "Homework is essential for reinforcing classroom learning.",
        "claim2": "Excessive homework contributes to student stress and sleep deprivation.",
        "expected": "Attack"
    },
    {
        "claim1": "Online learning is more effective than traditional classroom education.",
        "claim2": "Students in online courses often have lower completion rates than in-person classes.",
        "expected": "Attack"
    },
    {
        "claim1": "Banning smartphones in classrooms improves learning outcomes.",
        "claim2": "Studies show that students perform better academically when phones are restricted.",
        "expected": "Support"
    },
    {
        "claim1": "College degrees are essential for career success.",
        "claim2": "Many high-paying jobs now value skills and experience over formal degrees.",
        "expected": "Attack"
    },

    # Society
    {
        "claim1": "Universal basic income would eliminate poverty.",
        "claim2": "A guaranteed income could lead to inflation and reduced work incentives.",
        "expected": "Attack"
    },
    {
        "claim1": "Social media platforms should censor misinformation.",
        "claim2": "Censorship by private companies threatens freedom of expression.",
        "expected": "Attack"
    },
    {
        "claim1": "Legalizing all drugs would reduce drug-related violence.",
        "claim2": "Portugal's drug decriminalization policy has led to lower addiction rates and overdose deaths.",
        "expected": "Support"
    },
    {
        "claim1": "Immigration strengthens the economy of host countries.",
        "claim2": "Immigrants contribute to economic growth through taxes and workforce participation.",
        "expected": "Support"
    },
    {
        "claim1": "The death penalty effectively deters crime.",
        "claim2": "Research shows no significant difference in crime rates between states with and without the death penalty.",
        "expected": "Attack"
    },

    # Economy
    {
        "claim1": "Minimum wage increases help reduce income inequality.",
        "claim2": "Higher minimum wages often lead to job losses in small businesses.",
        "expected": "Attack"
    },
    {
        "claim1": "Free trade agreements benefit all participating countries.",
        "claim2": "Free trade often leads to job losses in developed countries' manufacturing sectors.",
        "expected": "Attack"
    },
    {
        "claim1": "Cryptocurrencies will replace traditional fiat currencies.",
        "claim2": "The volatility of cryptocurrencies makes them unsuitable as stable currencies.",
        "expected": "Attack"
    },
    {
        "claim1": "Universal healthcare is too expensive for governments to provide.",
        "claim2": "Countries with universal healthcare spend less per capita on health than the US.",
        "expected": "Attack"
    },
    {
        "claim1": "Tax cuts for the wealthy stimulate economic growth.",
        "claim2": "Evidence shows tax cuts often increase inequality without boosting long-term growth.",
        "expected": "Attack"
    },

    # Science
    {
        "claim1": "Genetically modified crops are safe for human consumption.",
        "claim2": "Over 2,000 studies confirm that GMOs are as safe as conventional crops.",
        "expected": "Support"
    },
    {
        "claim1": "Climate change is primarily caused by natural cycles, not human activity.",
        "claim2": "97% of climate scientists agree that human activities are the dominant cause of global warming.",
        "expected": "Attack"
    },
    {
        "claim1": "Vaccines cause autism in children.",
        "claim2": "Numerous large-scale studies have found no link between vaccines and autism.",
        "expected": "Attack"
    },
    {
        "claim1": "Space exploration is a waste of resources.",
        "claim2": "Space research drives technological innovations with benefits on Earth.",
        "expected": "Attack"
    },
    {
        "claim1": "Alternative medicine is as effective as conventional medicine.",
        "claim2": "Scientific reviews find no evidence that homeopathy or similar practices outperform placebos.",
        "expected": "Attack"
    },

    # Politics
    {
        "claim1": "Democracy is the most effective form of government.",
        "claim2": "Democratic systems often suffer from slow decision-making and political gridlock.",
        "expected": "Attack"
    },
    {
        "claim1": "Government surveillance is necessary for national security.",
        "claim2": "Mass surveillance violates citizens' privacy rights and can be abused.",
        "expected": "Attack"
    },
    {
        "claim1": "Term limits for politicians improve government effectiveness.",
        "claim2": "Term limits remove experienced leaders and increase lobbyist influence.",
        "expected": "Attack"
    },
    {
        "claim1": "Direct democracy ensures better representation of citizens' will.",
        "claim2": "Frequent referendums can oversimplify complex issues and lead to populist outcomes.",
        "expected": "Attack"
    },
    {
        "claim1": "International organizations like the UN are essential for global peace.",
        "claim2": "The UN provides a platform for diplomacy and conflict resolution worldwide.",
        "expected": "Support"
    },

    # Culture
    {
        "claim1": "Video games are a form of artistic expression.",
        "claim2": "Many video games tell complex stories and explore deep philosophical themes.",
        "expected": "Support"
    },
    {
        "claim1": "Censorship of art is sometimes necessary to protect society.",
        "claim2": "Artistic freedom is essential for cultural progress and social critique.",
        "expected": "Attack"
    },
    {
        "claim1": "Social media has improved human communication.",
        "claim2": "Social media has led to increased loneliness and superficial relationships.",
        "expected": "Attack"
    },
    {
        "claim1": "Reading fiction improves empathy and social understanding.",
        "claim2": "Psychological studies link reading novels to higher emotional intelligence.",
        "expected": "Support"
    },
    {
        "claim1": "Popular music today lacks depth compared to older generations.",
        "claim2": "Music complexity and lyrical depth vary across eras, not strictly by generation.",
        "expected": "Attack"
    },

    # Future
    {
        "claim1": "Humans will colonize Mars within the next 50 years.",
        "claim2": "Current technology and radiation risks make Mars colonization unlikely in the near future.",
        "expected": "Attack"
    },
    {
        "claim1": "Artificial general intelligence will surpass human intelligence by 2050.",
        "claim2": "Most AI experts believe we're decades away from achieving human-level AI.",
        "expected": "Attack"
    },
    {
        "claim1": "Renewable energy will completely replace fossil fuels by 2040.",
        "claim2": "Current energy storage limitations make a complete transition unlikely by 2040.",
        "expected": "Attack"
    },
    {
        "claim1": "Climate refugees will become one of the biggest humanitarian crises of the century.",
        "claim2": "Rising sea levels could displace hundreds of millions of people by 2100.",
        "expected": "Support"
    },
    {
        "claim1": "Human life expectancy will surpass 120 years this century.",
        "claim2": "Despite medical advances, biological limits make such lifespans unlikely for most people.",
        "expected": "Attack"
    }
]