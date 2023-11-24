from web3 import Web3

def get_price_feed(rpc, address):
    web3 = Web3(Web3.HTTPProvider(rpc))

    # AggregatorV3Interface ABI
    abi = '[{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'
    
    # Price Feed address
    address = Web3.to_checksum_address(address)

    # Set up contract instance
    contract = web3.eth.contract(address=address, abi=abi)

    # Make call to latestRoundData()
    latestData = contract.functions.latestRoundData().call()
    price = latestData[1]
    return price

if __name__ == "__main__":
    price = get_price_feed(
        rpc='https://rpc.ankr.com/eth',
        address='0x449d117117838ffa61263b61da6301aa2a88b13a'
    )

    print(price)